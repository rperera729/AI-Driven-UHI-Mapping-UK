# AI-Driven Urban Heat Island Mapping for UK Climate Resilience

This research project implements a Deep Learning architecture (U-Net) to perform thermal super-resolution and geospatial inpainting across the Greater London Authority (GLA). The system fuses multi-sensor satellite data to reconstruct high-resolution (100m) surface temperatures from coarse-resolution (1km) daily observations.

## Project Status: Optimized and Validated
The research pipeline is finalized. The model has undergone a 100-epoch optimization cycle, achieving stable convergence. The system is now capable of identifying neighborhood-level thermal intensity with high spatial fidelity, even in the presence of partial cloud contamination.

## Technical Performance Metrics
* **Architecture**: 5-Layer Multi-Sensor U-Net
* **Optimization Strategy**: Hybrid L1-L2 (MSE) Loss (70/30 weighting)
* **Generalization Accuracy**: ±4.85°C RMSE (Validated on an independent 10% test partition)
* **Numerical Convergence**: Normalized RMSE of 0.15, demonstrating robust spatial pattern recognition

## Multi-Sensor Fusion Strategy
The model synchronizes three distinct data streams onto the British National Grid (EPSG:27700):
* **Landsat 9 (TIRS-2)**: High-resolution Ground Truth reference (100m)
* **Sentinel-3 (SLSTR)**: Daily Coarse Thermal Input (1km) for temporal continuity
* **NDVI (Vegetation Index)**: To account for the cooling effects of urban green infrastructure
* **SRTM DEM**: Digital Elevation Model for topographic temperature correction
* **QA Cloud Mask**: Automated guidance for neural inpainting in contaminated zones

## Technical Workflow
1. **Data Engineering**: Automated Python scripts for temporal scene matching and radiometric calibration.
2. **Feature Stacking**: Generation of 616 multi-modal tensors (128x128 pixel patches).
3. **Neural Optimization**: 100-epoch training utilizing Tesla T4 GPU acceleration and a ReduceLROnPlateau scheduler.
4. **Radiometric Validation**: Metadata-synchronized audit to perform the inverse transformation from normalized tensors to physical Celsius units.

## Repository Structure
* `model.py`: Neural network architecture and layer definitions.
* `batch_stacker.py`: Geospatial data engineering and multi-sensor fusion logic.
* `UHI_Mapping_Full_Pipeline.ipynb`: Primary research notebook containing validated results.
* `requirements.txt`: Environment dependencies for system reproducibility.

## Policy and Planning Application
This tool is engineered for Urban Planning and Climate Resilience. By integrating results with London Borough Administrative Boundaries, the project provides local authorities with high-resolution thermal intelligence required to prioritize neighborhood-level interventions, such as cool-roof installations and strategic urban greening.
