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

# OFFICIAL GREATER LONDON BOUNDARY (EPSG:27700 - British National Grid)
# [min_x, min_y, max_x, max_y]
GLA_BBOX = (503000, 155000, 560000, 201000)

def run_batch_stacking():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for date in MATCHED_DATES:
        try:
            print(f"--- Processing Date: {date} ---")
            
            # 1. FIND FILES (Get the first string from the list using [0])
            ls_b10_list = glob.glob(f"{BASE_DIR}/landsat/**/*{date}*_ST_B10.TIF", recursive=True)
            ls_b4_list  = glob.glob(f"{BASE_DIR}/landsat/**/*{date}*_SR_B4.TIF", recursive=True)
            ls_b5_list  = glob.glob(f"{BASE_DIR}/landsat/**/*{date}*_SR_B5.TIF", recursive=True)

            if not (ls_b10_list and ls_b4_list and ls_b5_list):
                print(f"Skipping {date}: Missing bands.")
                continue

            s3_folders = glob.glob(f"{BASE_DIR}/sentinel/*{date}*.SEN3")
            if not s3_folders: continue
            s3_nc = os.path.join(s3_folders[0], "LST_in.nc")

            # 2. LOAD & CLIP LANDSAT (The Master Reference)
            # We clip immediately to prevent the 'Unable to allocate GiB' error
            ls_raw = rioxarray.open_rasterio(ls_b10_list[0]).rio.reproject("EPSG:27700")
            ls_raw = ls_raw.rio.clip_box(*GLA_BBOX)
            
            # 3. CALIBRATE (Celsius)
            ls_c = (ls_raw.astype(float) * 0.00341802) + 149.0 - 273.15
            
            # 4. HARMONIZE NDVI
            red = rioxarray.open_rasterio(ls_b4_list[0]).rio.reproject_match(ls_raw).astype(float)
            nir = rioxarray.open_rasterio(ls_b5_list[0]).rio.reproject_match(ls_raw).astype(float)
            ndvi = (nir - red) / (nir + red + 1e-10)
            
            # 5. HARMONIZE SENTINEL
            s3_ds = xr.open_dataset(s3_nc)
            s3_lst = s3_ds['LST'].rename({'rows': 'y', 'columns': 'x'}).rio.write_crs("EPSG:4326")
            s3_matched = s3_lst.rio.reproject_match(ls_raw, resampling=Resampling.bilinear)
            s3_c = s3_matched - 273.15 if s3_matched.max() > 100 else s3_matched
            
            # 6. HARMONIZE DEM
            dem = rioxarray.open_rasterio(DEM_PATH).rio.reproject_match(ls_raw, resampling=Resampling.bilinear)

            # 7. CLEAN AND STACK
            ls_c = ls_c.squeeze().drop_vars("band", errors="ignore")
            s3_c = s3_c.squeeze().drop_vars("band", errors="ignore")
            ndvi = ndvi.squeeze().drop_vars("band", errors="ignore")
            dem = dem.squeeze().drop_vars("band", errors="ignore")

            final_stack = xr.concat([ls_c, s3_c, ndvi, dem], dim="band")
            
            output_file = os.path.join(OUTPUT_DIR, f"UHI_Stack_{date}.tif")
            final_stack.rio.to_raster(output_file)
            
            print(f"✅ Success: {output_file}")

        except Exception as e:
            print(f"❌ Error on {date}: {e}")

if __name__ == "__main__":
    run_batch_stacking()
