import rioxarray
import xarray as xr
import os
import glob
import numpy as np
from rasterio.enums import Resampling

# --- CONFIGURATION ---
BASE_DIR = r"D:\UHI_Project\data_raw"
OUTPUT_DIR = r"D:\UHI_Project\data_processed"
DEM_PATH = os.path.join(BASE_DIR, "srtm_dem_30m.tif")
MATCHED_DATES = ["20230805", "20230603", "20240629", "20240823", "20250710", "20250811"]

# Greater London Authority (GLA) Bounding Box (EPSG:27700)
GLA_BBOX = (503000, 155000, 560000, 201000)

def run_batch_stacking():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for date in MATCHED_DATES:
        try:
            print(f"--- Processing Date: {date} ---")
            
            # 1. FIND FILES
            ls_b10 = glob.glob(f"{BASE_DIR}/landsat/**/*{date}*_ST_B10.TIF", recursive=True)
            ls_b4  = glob.glob(f"{BASE_DIR}/landsat/**/*{date}*_SR_B4.TIF", recursive=True)
            ls_b5  = glob.glob(f"{BASE_DIR}/landsat/**/*{date}*_SR_B5.TIF", recursive=True)
            ls_qa  = glob.glob(f"{BASE_DIR}/landsat/**/*{date}*_QA_PIXEL.TIF", recursive=True)
            s3_folders = glob.glob(f"{BASE_DIR}/sentinel/*{date}*.SEN3")

            if not (ls_b10 and ls_b4 and ls_b5 and ls_qa and s3_folders):
                print(f"Skipping {date}: Missing bands or QA file.")
                continue

            # 2. LANDSAT LST & QA MASK
            ls_raw = rioxarray.open_rasterio(ls_b10[0]).rio.reproject("EPSG:27700").rio.clip_box(*GLA_BBOX)
            # Calibration to Celsius
            ls_c = (ls_raw.astype(float) * 0.00341802) + 149.0 - 273.15
            
            # QA Mask (1 = Clear, 0 = Cloud/Shadow)
            qa = rioxarray.open_rasterio(ls_qa[0]).rio.reproject_match(ls_raw)
            mask = xr.where((qa == 21824) | (qa == 21952), 1, 0)

            # 3. NDVI
            red = rioxarray.open_rasterio(ls_b4[0]).rio.reproject_match(ls_raw).astype(float)
            nir = rioxarray.open_rasterio(ls_b5[0]).rio.reproject_match(ls_raw).astype(float)
            ndvi = (nir - red) / (nir + red + 1e-10)
            
            # 4. SENTINEL LST
            s3_nc = os.path.join(s3_folders[0], "LST_in.nc")
            s3_ds = xr.open_dataset(s3_nc)
            s3_lst = s3_ds['LST'].rename({'rows': 'y', 'columns': 'x'}).rio.write_crs("EPSG:4326")
            s3_matched = s3_lst.rio.reproject_match(ls_raw, resampling=Resampling.bilinear)
            s3_c = s3_matched - 273.15 if s3_matched.max() > 100 else s3_matched
            
            # 5. DEM
            dem = rioxarray.open_rasterio(DEM_PATH).rio.reproject_match(ls_raw, resampling=Resampling.bilinear)

            # 6. UNIFY COORDINATES (Squeeze and Drop)
            ls_c = ls_c.squeeze().drop_vars("band", errors="ignore")
            s3_c = s3_c.squeeze().drop_vars("band", errors="ignore")
            ndvi = ndvi.squeeze().drop_vars("band", errors="ignore")
            dem = dem.squeeze().drop_vars("band", errors="ignore")
            mask = mask.squeeze().drop_vars("band", errors="ignore")

            # 7. STACK [Landsat, Sentinel, NDVI, DEM, QA_Mask]
            final_stack = xr.concat([ls_c, s3_c, ndvi, dem, mask], dim="band")
            final_stack = final_stack.assign_coords(band=[1, 2, 3, 4, 5])
            
            output_file = os.path.join(OUTPUT_DIR, f"UHI_Stack_{date}.tif")
            final_stack.rio.to_raster(output_file, dtype="float32")
            
            print(f" Success! 5-Layer Stack saved: {output_file}")

        except Exception as e:
            print(f" Error on {date}: {e}")

if __name__ == "__main__":
    run_batch_stacking()
