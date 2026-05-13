import os
import glob
from rasterio.merge import merge
import rasterio

# Configuration paths
DEM_FOLDER = r"D:\UHI_Project\DEM"
OUTPUT_FOLDER = r"D:\UHI_Project\data_raw"
OUTPUT_FILENAME = "srtm_dem_30m.tif"

# 1. Find all TIFF tiles in the DEM folder
dem_tiles = glob.glob(os.path.join(DEM_FOLDER, "*.tif"))
if not dem_tiles:
    print(f"No TIFF files found in {DEM_FOLDER}")
    exit()

# 2. Open and merge
src_files_to_mosaic = []
for fp in dem_tiles:
    src = rasterio.open(fp)
    src_files_to_mosaic.append(src)

mosaic, out_trans = merge(src_files_to_mosaic)

# 3. Save the result
out_meta = src_files_to_mosaic[0].meta.copy()
out_meta.update({
    "height": mosaic.shape[1],
    "width": mosaic.shape[2],
    "transform": out_trans
})

out_path = os.path.join(OUTPUT_FOLDER, OUTPUT_FILENAME)
with rasterio.open(out_path, "w", **out_meta) as dest:
    dest.write(mosaic)

print(f" DEM tiles merged into '{out_path}'")
