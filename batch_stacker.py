import rioxarray
import xarray as xr
import os
import glob
from rasterio.enums import Resampling

# --- CONFIGURATION ---
BASE_DIR = r"D:\UHI_Project\data_raw"
OUTPUT_DIR = r"D:\UHI_Project\data_processed"
DEM_PATH = os.path.join(BASE_DIR, "srtm_dem_30m.tif")
MATCHED_DATES = ["20230805", "20230603", "20240629", "20240823", "20250710", "20250811"]

# Greater London Bounding Box (British National Grid EPSG:27700)
# This prevents "Unable to allocate MiB" errors by focusing only on London
LONDON_BBOX = (500000, 160000, 560000, 205000) # (minx, miny, maxx, maxy)

def run_batch_stacking():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for date in MATCHED_DATES:
        try:
            print(f"\n--- Processing Date: {date} ---")
            
            # 1. FIND LANDSAT FILES
            ls_b10 = glob.glob(f"{BASE_DIR}/landsat/**/*{date}*_ST_B10.TIF", recursive=True)
            ls_b4  = glob.glob(f"{BASE_DIR}/landsat/**/*{date}*_SR_B4.TIF", recursive=True)
            ls_b5  = glob.glob(f"{BASE_DIR}/landsat/**/*{date}*_SR_B5.TIF", recursive=True)

            if not (ls_b10 and ls_b4 and ls_b5):
                print(f" Missing Landsat bands for {date}. Skipping.")
                continue

            # 2. FIND SENTINEL FILE
            s3_folders = glob.glob(f"{BASE_DIR}/sentinel/*{date}*.SEN3")
            if not s3_folders:
                print(f" Sentinel folder not found for {date}. Skipping.")
                continue
            s3_nc = os.path.join(s3_folders[0], "LST_in.nc")

            # --- PROCESSING LANDSAT ---
            # Load, Reproject, and CLIP TO LONDON to save memory
            ls = rioxarray.open_rasterio(ls_b10[0]).rio.reproject("EPSG:27700")
            ls = ls.rio.clip_box(*LONDON_BBOX)
            ls_c = (ls * 0.00341802) + 149.0 - 273.15
            
            red = rioxarray.open_rasterio(ls_b4[0]).rio.reproject_match(ls)
            nir = rioxarray.open_rasterio(ls_b5[0]).rio.reproject_match(ls)
            ndvi = (nir - red) / (nir + red)
            
            # --- PROCESSING SENTINEL ---
            s3_ds = xr.open_dataset(s3_nc)
            s3_lst = s3_ds['LST'].rename({'rows': 'y', 'columns': 'x'})
            s3_lst = s3_lst.rio.write_crs("EPSG:4326")
            s3_matched = s3_lst.rio.reproject_match(ls, resampling=Resampling.bilinear)
            s3_c = s3_matched - 273.15
            
            # --- PROCESSING DEM ---
            dem = rioxarray.open_rasterio(DEM_PATH).rio.reproject_match(ls, resampling=Resampling.bilinear)

            # --- FIXING COORDINATES FOR STACKING ---
            # Remove 'band' from Landsat/DEM so they match Sentinel (which has no band)
            ls_c = ls_c.squeeze().drop_vars("band", errors="ignore")
            s3_c = s3_c.squeeze().drop_vars("band", errors="ignore")
            ndvi = ndvi.squeeze().drop_vars("band", errors="ignore")
            dem = dem.squeeze().drop_vars("band", errors="ignore")

            # --- STACK AND SAVE ---
            final_stack = xr.concat([ls_c, s3_c, ndvi, dem], dim="band")
            final_stack = final_stack.assign_coords(band=[1, 2, 3, 4])
            
            output_file = os.path.join(OUTPUT_DIR, f"UHI_Stack_{date}.tif")
            final_stack.rio.to_raster(output_file)
            
            print(f" Success! Harmonized stack saved: {output_file}")

        except Exception as e:
            print(f" Error on {date}: {e}")

if __name__ == "__main__":
    run_batch_stacking()
