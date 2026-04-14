This project utilizes deep learning (U-Net) to map Urban Heat Island (UHI) intensity across the UK, focusing on the 2023-2025 summer periods.

## Project Status: Preprocessing Phase
Currently harmonizing multi-sensor satellite data to a unified 100m grid (EPSG:27700).

### Data Sources
* **Landsat 8 (Level 2):** High-resolution Thermal (100m) & NDVI data.
* **Sentinel-3 (SLSTR):** Coarse LST (1km) for temporal gap-filling.
* **SRTM DEM:** 30m Digital Elevation Model for altitude-based temperature correction.

### Technical Workflow
1. **Spatial Alignment:** Reprojecting all sensors to the British National Grid.
2. **Feature Engineering:** Calculating NDVI and LST Scaling.
3. **AI Training:** Implementing a Partial Convolution U-Net for cloud inpainting.
