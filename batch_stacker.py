import os
import glob
import numpy as np
import xarray as xr
import rioxarray
from rasterio.enums import Resampling
from scipy.interpolate import griddata
import pyproj

# ---------------- CONFIG ---------------- #
BASE_DIR = r"D:\UHI_Project\data_raw"
OUTPUT_DIR = r"D:\UHI_Project\data_processed"
DEM_PATH = os.path.join(BASE_DIR, "srtm_dem_30m.tif")

MATCHED_DATES = [
    "20230805",
    "20230603",
    "20240629",
    "20240823",
    "20250710"
]

GLA_BBOX = (503000, 155000, 560000, 201000)
# ---------------------------------------- #


# ---------------- NORMALISATION ---------------- #
def simple_norm(x):
    vals = x.values.astype("float32")
    v_min, v_max = np.nanpercentile(vals, 2), np.nanpercentile(vals, 98)

    if not np.isfinite(v_min) or v_max <= v_min:
        return xr.zeros_like(x)

    return ((x - v_min) / (v_max - v_min)).clip(0, 1)


# ---------------- SAFE TIFF LOADER ---------------- #
def safe_open_tif(path, match=None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

    da = rioxarray.open_rasterio(path, masked=True)

    if match is not None:
        da = da.rio.reproject_match(match)

    return da


# ---------------- SENTINEL LOADER (FIXED) ---------------- #
def load_sentinel(s3_folder):
    s3_nc = os.path.join(s3_folder, "LST_in.nc")

    if not os.path.exists(s3_nc):
        raise FileNotFoundError(f"Missing Sentinel LST file: {s3_nc}")

    ds = xr.open_dataset(s3_nc)
    lst = ds["LST"] - 273.15

    geo_files = (
        glob.glob(os.path.join(s3_folder, "*geo*.nc")) +
        glob.glob(os.path.join(s3_folder, "*tie*.nc")) +
        glob.glob(os.path.join(s3_folder, "*geoloc*.nc"))
    )

    if not geo_files:
        raise FileNotFoundError("Missing Sentinel geolocation file")

    geo = xr.open_dataset(geo_files[0])

    lat_name = next((v for v in geo.variables if "lat" in v.lower()), None)
    lon_name = next((v for v in geo.variables if "lon" in v.lower()), None)

    if lat_name is None or lon_name is None:
        raise KeyError("Latitude/Longitude not found in Sentinel geo file")

    lat = geo[lat_name]
    lon = geo[lon_name]

    return lst, lat, lon


# ---------------- MAIN PIPELINE ---------------- #
def run_batch_stacking():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for date in MATCHED_DATES:
        try:
            print(f"\n--- Processing {date} ---")

            # ---------- FIND FILES ---------- #
            ls_b10 = glob.glob(f"{BASE_DIR}/landsat/**/*{date}*_ST_B10.TIF", recursive=True)
            ls_b4  = glob.glob(f"{BASE_DIR}/landsat/**/*{date}*_SR_B4.TIF", recursive=True)
            ls_b5  = glob.glob(f"{BASE_DIR}/landsat/**/*{date}*_SR_B5.TIF", recursive=True)
            ls_qa  = glob.glob(f"{BASE_DIR}/landsat/**/*{date}*_QA_PIXEL.TIF", recursive=True)
            s3_dir = glob.glob(f"{BASE_DIR}/sentinel/*{date}*.SEN3")

            if not (ls_b10 and ls_b4 and ls_b5 and ls_qa and s3_dir):
                print("[SKIP] Missing required files")
                continue

            # ---------- LANDSAT LST ---------- #
            ls = safe_open_tif(ls_b10[0]) \
                .rio.reproject("EPSG:27700") \
                .rio.clip_box(*GLA_BBOX)

            ls_c = (ls.where(ls > 0) * 0.00341802 + 149.0) - 273.15

            # ---------- CLOUD MASK ---------- #
            qa = safe_open_tif(ls_qa[0], match=ls).astype("uint16")
            cloud = (qa & (1 << 3)) > 0
            mask = (~cloud).astype(int)

            ls_c = ls_c.where(mask == 1)

            # ---------- NDVI ---------- #
            red = safe_open_tif(ls_b4[0], match=ls).astype("float32")
            nir = safe_open_tif(ls_b5[0], match=ls).astype("float32")

            red = red * 0.0000275 - 0.2
            nir = nir * 0.0000275 - 0.2

            ndvi = (nir - red) / (nir + red + 1e-6)
            ndvi = ndvi.clip(-1, 1).where(mask == 1)

            # ---------- SENTINEL ---------- #
            lst, lat, lon = load_sentinel(s3_dir[0])

            lst_vals = lst.values.flatten()
            lat_vals = lat.values.flatten()
            lon_vals = lon.values.flatten()

            valid = np.isfinite(lst_vals) & np.isfinite(lat_vals) & np.isfinite(lon_vals)

            lst_vals = lst_vals[valid]
            lat_vals = lat_vals[valid]
            lon_vals = lon_vals[valid]

            lon_grid, lat_grid = np.meshgrid(ls.x.values, ls.y.values)

            transformer = pyproj.Transformer.from_crs(
                "EPSG:27700", "EPSG:4326", always_xy=True
            )

            lon_t, lat_t = transformer.transform(lon_grid, lat_grid)

            s3_interp = griddata(
                (lon_vals, lat_vals),
                lst_vals,
                (lon_t, lat_t),
                method="linear"
            )

            s3_c = xr.DataArray(
                s3_interp,
                coords={"y": ls.y, "x": ls.x},
                dims=["y", "x"]
            ).where(mask.squeeze() == 1)

            # ---------- DEM ---------- #
            dem = safe_open_tif(DEM_PATH) \
                .rio.reproject_match(ls, resampling=Resampling.bilinear)

            # ---------- NORMALISE ---------- #
            def sq(x):
                return x.squeeze(drop=True)

            norm_ls   = simple_norm(sq(ls_c))
            norm_s3   = simple_norm(sq(s3_c))
            norm_ndvi = simple_norm(sq(ndvi))
            norm_dem  = simple_norm(sq(dem))
            mask_2d   = sq(mask)

            # ---------- STACK (FIXED BAND ISSUE) ---------- #
            stack = xr.concat(
                [
                    norm_ls.expand_dims(band=[1]),
                    norm_s3.expand_dims(band=[2]),
                    norm_ndvi.expand_dims(band=[3]),
                    norm_dem.expand_dims(band=[4]),
                    mask_2d.expand_dims(band=[5]),
                ],
                dim="band"
            )

            # ---------- SAVE ---------- #
            out_path = os.path.join(OUTPUT_DIR, f"UHI_Stack_{date}.tif")
            stack.rio.to_raster(out_path, dtype="float32")

            print("[OK] Saved:", out_path)

        except Exception as e:
            print(f"[FAILED DATE {date}]")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    run_batch_stacking()