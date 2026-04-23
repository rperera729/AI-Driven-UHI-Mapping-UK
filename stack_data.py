import rioxarray
import xarray as xr
from rasterio.enums import Resampling
import os

def create_stack(date_str, ls_b10, ls_b4, ls_b5, s3_nc, dem_path, output_dir):
    print(f"--- Stacking Data for {date_str} ---")
    
    # 1. Load Landsat LST and scale to Celsius
    ls_lst = rioxarray.open_rasterio(ls_b10).rio.reproject("EPSG:27700")
    ls_lst_c = (ls_lst * 0.00341802) + 149.0 - 273.15
    
    # 2. Calculate NDVI (Vegetation)
    red = rioxarray.open_rasterio(ls_b4).rio.reproject_match(ls_lst)
    nir = rioxarray.open_rasterio(ls_b5).rio.reproject_match(ls_lst)
    ndvi = (nir - red) / (nir + red)
    
    # 3. Align Sentinel-3 (Upsample to 100m)
    s3_ds = xr.open_dataset(s3_nc)
    s3_lst = s3_ds['LST'].rio.write_crs("EPSG:4326")
    s3_aligned = s3_lst.rio.reproject_match(ls_lst, resampling=Resampling.bilinear)
    s3_aligned_c = s3_aligned - 273.15 # Kelvin to Celsius
    
    # 4. Align DEM
    dem = rioxarray.open_rasterio(dem_path).rio.reproject_match(ls_lst, resampling=Resampling.bilinear)
    
    # 5. Stack and Save
    stack = xr.concat([ls_lst_c, s3_aligned_c, ndvi, dem], dim="band")
    stack.rio.to_raster(os.path.join(output_dir, f"Stack_{date_str}.tif"))
    print(f" Saved Stacked Image for {date_str}")

# Usage: Run this for your 6 dates!
