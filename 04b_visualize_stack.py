import rioxarray
import matplotlib.pyplot as plt

STACK_PATH = r"D:\UHI_Project\data_processed\UHI_Stack_20250629.tif"

def plot_layer(ax, data, title, cmap):
    im = ax.imshow(data, cmap=cmap)
    ax.set_title(title)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

def visualize():
    ds = rioxarray.open_rasterio(STACK_PATH)

    print("Stack shape:", ds.shape)

    landsat = ds[0].values
    sentinel = ds[1].values
    ndvi = ds[2].values
    dem = ds[3].values
    mask = ds[4].values

    fig, axes = plt.subplots(1, 5, figsize=(20, 5))

    plot_layer(axes[0], landsat, "Landsat", "inferno")
    plot_layer(axes[1], sentinel, "Sentinel", "inferno")
    plot_layer(axes[2], ndvi, "NDVI", "YlGn")
    plot_layer(axes[3], dem, "DEM", "terrain")
    plot_layer(axes[4], mask, "Mask", "gray")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize()