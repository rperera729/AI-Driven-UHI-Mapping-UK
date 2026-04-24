# AI-Driven Urban Heat Island Mapping for UK Climate Resilience

This project utilizes Deep Learning (Partial Convolution U-Net) to map and predict Urban Heat Island (UHI) intensity across Greater London. By fusing multi-sensor satellite data, we reconstruct high-resolution surface temperatures even in the presence of cloud cover.

##  Project Status: Data Harmonization Complete
We have successfully implemented an automated preprocessing pipeline that synchronizes three distinct data streams into a unified 100m spatial grid.

###  Data Sources
*   **Landsat 8 (TIRS/OLI):** High-resolution Surface Temperature (100m) and multispectral bands for NDVI calculation.
*   **Sentinel-3 (SLSTR):** Daily Land Surface Temperature (1km) used for temporal continuity and AI fusion.
*   **SRTM DEM:** 30m Digital Elevation Model used for topographic temperature correction (Lapse Rate analysis).

###  Technical Workflow
1.  **Scene Matching:** Recursive Python scripts to identify perfect temporal overlaps between Landsat and Sentinel-3 (6 "Gold Standard" scenes identified for 2023-2025).
2.  **Spatial Harmonization:**
    *   Reprojection of all sensors to the **British National Grid (EPSG:27700)**.
    *   Bilinear Resampling of 1km Sentinel data to 100m Landsat grid.
    *   Spatial Mosaicking of SRTM tiles for a continuous Greater London baseline.
3.  **Feature Engineering:** 
    *   Radiometric calibration of Thermal bands to Celsius.
    *   Calculation of NDVI to account for the "Urban Green Cooling" effect.
4.  **AI Implementation (In Progress):**
    *   Patch-based learning using 128x128 pixel windows.
    *   Training a **Partial Convolution U-Net** for intelligent cloud inpainting.

##  Repository Structure
*   `scripts/`: Contains Python automation for data matching, stacking, and visualization.
*   `data/`: (Ignored via .gitignore) Local storage for raw .TIF and .nc files.
*   `data_processed/`: Harmonized 4-band tensors ready for AI training.

##  Setup & Requirements
The project is developed using **Python 3.11.9**. Key dependencies include:
*   `rioxarray` & `xarray` (Geospatial data handling)
*   `rasterio` (Raster processing)
*   `PyTorch` (AI Model Architecture)

---
*Developed as part of the UK Climate Resilience research collaboration.*
