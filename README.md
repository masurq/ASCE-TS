ASCE-TS: An Asynchronous Optical-SAR Spatiotemporal Context-Enhanced Framework for Limited-Sample Long-Term Coastal Wetland Mapping
This repository is deliberately paper-faithful. Values, thresholds, candidate ranges, cloud-mask details, neighborhood size, TWDTW temporal-penalty function, RFE elimination schedule, and cascade-forest hyperparameters that are not numerically specified in the manuscript are not invented here. They are exposed as required configuration fields and marked `PAPER_NOT_SPECIFIED`.
Implemented pipeline
Sentinel-1 / Sentinel-2 feature construction.
Native-timestamp asynchronous optical–SAR representation.
SAR-guided reconstruction of missing optical observations:
local SAR trajectory: 3 observations before + 6 after;
DTW matching to historical states having valid optical observations;
similarity × temporal-proximity weighting;
`sigma_d`: median historical DTW distance;
`sigma_r`: median temporal interval of available observations.
Similarity-guided multidimensional spatiotemporal context:
spatial kernel;
temporal trajectory-consistency kernel;
Fisher-score feature discriminability;
joint contribution weights.
Within-class sample refinement.
TWDTW trajectory distance and weighted k-medoids prototypes.
Prototype-distance features.
Random-forest-based recursive feature elimination.
Cascade-forest classification.
OA, Cohen's Kappa and macro-F1 evaluation.
Progressive ablation and training-sample-size experiment entry points.
Paper feature set
Sentinel-1: `VV_lin, VH_lin, VV_dB, VH_dB, RVI, H, A, alpha`
Sentinel-2: `B2, B3, B4, B5, B6, B7, B8, B11, B12, NDVI, NDWI, NDBI, SAVI, EVI, MSAVI, NDMI, NBR, B8_B4`
The manuscript states that S1 GRD IW VV/VH data use GEE-provided orbit update, thermal-noise removal, radiometric calibration and terrain correction, followed by low-quality edge masking and Refined Lee filtering. S2 Level-2A data are cloud/cloud-shadow/low-quality masked; 20-m red-edge/SWIR bands are resampled to 10 m.
