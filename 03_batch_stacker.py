import os
import glob
import numpy as np
import xarray as xr
import rioxarray
from rasterio.enums import Resampling
from scipy.interpolate import griddata
import pyproj
from rioxarray.merge import merge_arrays

# ---------------- CONFIG ---------------- #
BASE_DIR = r"D:\UHI_Project\data_raw"
OUTPUT_DIR = r"D:\UHI_Project\data_processed"
DEM_PATH = os.path.join(BASE_DIR, "srtm_dem_30m.tif")

MATCHED_DATES = ["20250629"]
#MATCHED_DATES = ["20230616", "20230702", "20240602", "20250707", "20250824", "20230616", "20240602", "20250707", "20250831", "20230826", "20250815", "20250831", "20230614", "20230817", "20250619", "20230811", "20240829", "20250629", "20230608", "20230811", "20240626", "20240829", "20250629", "20230615", "20240601", "20230615", "20240617", "20240820", "20250620", "20230622", "20240608", "20240811", "20250713"]


# GLA_BBOX = (503000, 155000, 560000, 201000)
# Glasgow City RegionBounding Box (EPSG:27700)
GLA_BBOX = (218808, 600859, 312369, 690979)
# ---------------------------------------- #

# --- MASTER GRID SETUP ---
# Calculate a common 30m grid aligned to the bounding box so all mosaiced tiles snap perfectly without grid gaps.
_minx, _miny, _maxx, _maxy = GLA_BBOX
_minx = (_minx // 30) * 30
_miny = (_miny // 30) * 30
_maxx = (_maxx // 30 + 1) * 30
_maxy = (_maxy // 30 + 1) * 30

_x = np.arange(_minx + 15, _maxx, 30)
_y = np.arange(_maxy - 15, _miny, -30)

MASTER_GRID = xr.DataArray(
    np.zeros((1, len(_y), len(_x)), dtype=np.float32),
    coords={'band': [1], 'y': _y, 'x': _x},
    dims=['band', 'y', 'x']
).rio.write_crs('EPSG:27700')


# ---------------- NORMALISATION ---------------- #
def simple_norm(x):
    vals = x.values.astype("float32")
    
    # Check if the array is entirely NaNs to avoid the "All-NaN slice" warning
    if np.isnan(vals).all():
        return xr.zeros_like(x)
        
    v_min, v_max = np.nanpercentile(vals, 2), np.nanpercentile(vals, 98)

    if not np.isfinite(v_min) or v_max <= v_min:
        return xr.zeros_like(x)

    return ((x - v_min) / (v_max - v_min)).clip(0, 1)


# ---------------- MOSAIC & SNAP ---------------- #
def mosaic_files(file_list, match_xr=MASTER_GRID, resampling=Resampling.nearest):
    arrays = []
    for fp in file_list:
        try:
            da = rioxarray.open_rasterio(fp)
            da = da.rio.reproject_match(match_xr, resampling=resampling)
            arrays.append(da)
        except Exception:
            pass # Skip if outside bbox or other errors
            
    if not arrays:
        raise ValueError("No valid data found in bounding box.")
        
    return merge_arrays(arrays) if len(arrays) > 1 else arrays[0]


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
            s3_dir = glob.glob(f"{BASE_DIR}/sentinel/**/*{date}*.SEN3", recursive=True)

            if not (ls_b10 and ls_b4 and ls_b5 and ls_qa and s3_dir):
                print("[SKIP] Missing required files")
                continue

            # ---------- LANDSAT LST ---------- #
            ls = mosaic_files(ls_b10)

            ls_c = (ls.where(ls > 0) * 0.00341802 + 149.0) - 273.15

            # ---------- CLOUD MASK ---------- #
            qa = mosaic_files(ls_qa).astype("uint16")
            cloud = (qa & (1 << 3)) > 0
            
            # The mask should only be 1 if it's NOT cloudy AND if Landsat actually has data in that pixel
            mask = ((~cloud) & (ls > 0)).astype(int)

            ls_c = ls_c.where(mask == 1)

            # ---------- NDVI ---------- #
            red = mosaic_files(ls_b4).astype("float32")
            nir = mosaic_files(ls_b5).astype("float32")

            red = red * 0.0000275 - 0.2
            nir = nir * 0.0000275 - 0.2

            ndvi = (nir - red) / (nir + red + 1e-6)
            ndvi = ndvi.clip(-1, 1).where(mask == 1)

            # ---------- SENTINEL ---------- #
            # Combine all available Sentinel-3 scenes for the date
            lst_vals_list, lat_vals_list, lon_vals_list = [], [], []
            
            for s3_folder in s3_dir:
                try:
                    temp_lst, temp_lat, temp_lon = load_sentinel(s3_folder)
                    lst_vals_list.append(temp_lst.values.flatten())
                    lat_vals_list.append(temp_lat.values.flatten())
                    lon_vals_list.append(temp_lon.values.flatten())
                except Exception as e:
                    print(f"  [WARNING] Error loading Sentinel data from {s3_folder}: {e}")

            if not lst_vals_list:
                raise ValueError("No valid Sentinel data loaded for this date.")

            lst_vals = np.concatenate(lst_vals_list)
            lat_vals = np.concatenate(lat_vals_list)
            lon_vals = np.concatenate(lon_vals_list)

            # Create target grid in WGS84
            lon_grid, lat_grid = np.meshgrid(ls.x.values, ls.y.values)
            transformer = pyproj.Transformer.from_crs(
                "EPSG:27700", "EPSG:4326", always_xy=True
            )
            lon_t, lat_t = transformer.transform(lon_grid, lat_grid)

            # Dynamically filter Sentinel points using the target grid's bounds (+0.5 deg buffer)
            # This ensures griddata runs fast and adapts to any GLA_BBOX automatically
            min_lon, max_lon = lon_t.min() - 0.5, lon_t.max() + 0.5
            min_lat, max_lat = lat_t.min() - 0.5, lat_t.max() + 0.5

            valid = (
                np.isfinite(lst_vals) & np.isfinite(lat_vals) & np.isfinite(lon_vals) &
                (lon_vals >= min_lon) & (lon_vals <= max_lon) &
                (lat_vals >= min_lat) & (lat_vals <= max_lat)
            )

            lst_vals = lst_vals[valid]
            lat_vals = lat_vals[valid]
            lon_vals = lon_vals[valid]

            if len(lst_vals) == 0:
                print(f"  [WARNING] Date {date}: Sentinel points do not overlap the target area.")
                s3_interp = np.full(lon_t.shape, np.nan, dtype=np.float32)
            else:
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
                
            dem = dem.where(mask.squeeze() == 1)

            # ---------- NORMALISE ---------- #
            def sq(x):
                return x.squeeze(drop=True)

            ls_c_sq = sq(ls_c)
            s3_c_sq = sq(s3_c)
            ndvi_sq = sq(ndvi)
            dem_sq = sq(dem)

            # --- DEBUGGING: Check for empty layers before normalisation ---
            if np.isnan(ls_c_sq.values).all():
                print(f"  [WARNING] Date {date}: Landsat LST layer is entirely NaN (empty).")
            if np.isnan(s3_c_sq.values).all():
                print(f"  [WARNING] Date {date}: Sentinel LST layer is entirely NaN (empty).")
            if np.isnan(ndvi_sq.values).all():
                print(f"  [WARNING] Date {date}: NDVI layer is entirely NaN (empty).")
            if np.isnan(dem_sq.values).all():
                print(f"  [WARNING] Date {date}: DEM layer is entirely NaN (empty).")
                
            norm_ls   = simple_norm(ls_c_sq)
            norm_s3   = simple_norm(s3_c_sq)
            norm_ndvi = simple_norm(ndvi_sq)
            norm_dem  = simple_norm(dem_sq)
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