import os
from rasterio.merge import merge
import rasterio

# 1. List your two DEM tiles
dem_tiles = [
    r"D:\UHI_Project\data_raw\n51_e000_1arc_v3.tif",
    r"D:\UHI_Project\data_raw\n51_w001_1arc_v3.tif" # Update name if different
]

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

with rasterio.open(r"D:\UHI_Project\data_raw\srtm_dem_30m.tif", "w", **out_meta) as dest:
    dest.write(mosaic)

print(" DEM tiles merged into 'srtm_dem_30m.tif'")
