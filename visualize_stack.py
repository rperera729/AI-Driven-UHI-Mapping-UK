import rioxarray
import matplotlib.pyplot as plt
import os

# Path to the newly generated calibrated stack
stack_path = r"D:\UHI_Project\data_processed\UHI_Stack_20230805.tif"

if os.path.exists(stack_path):
    stack = rioxarray.open_rasterio(stack_path)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    titles = ['Landsat LST (C)', 'Sentinel LST (C)', 'NDVI (Veg)', 'DEM (Elev)']
    cmaps = ['magma', 'magma', 'YlGn', 'terrain']
    
    # These limits force the display to show the real data range
    # LST: 15-45C | NDVI: 0-0.6 | DEM: 0-150m
    limits = [(15, 45), (15, 45), (0, 0.6), (0, 150)]

    for i in range(4):
        layer = stack.sel(band=i+1)
        layer.plot(ax=axes[i], cmap=cmaps[i], add_labels=False, 
                   vmin=limits[i][0], vmax=limits[i][1])
        axes[i].set_title(titles[i])
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()
else:
    print(f"File not found. Please run batch_stacker.py first.")
