# Spatially Aware Deep Learning for Microclimate Prediction from High-Resolution Geospatial Imagery

_Idan Sulami, Alon Itzkovitch, Michael R. Kearney, Moni Shahar, and
Ofir Levy_

### Please contact Ofir Levy (levyofir@tauex.tau.ac.il) about the code or data

## Abstract:

Microclimate models are essential for linking climate to ecological processes, yet most physically based frameworks estimate temperature independently for each spatial unit and rely on simplified representations of lateral heat exchange. As a result, the spatial scales over which surrounding environmental conditions influence local microclimates remain poorly quantified, particularly in heterogeneous landscapes. Here, we show how remote sensing can help quantify the contribution of spatial context to microclimate temperature predictions. Building on convolutional neural network principles, we designed a task-specific deep neural network and trained a series of models in which the spatial extent of input data was systematically varied. Drone-derived spatial layers and meteorological data were used to predict ground temperature at a focal location, allowing direct assessment of how prediction accuracy changes with increasing spatial context. Our results show that incorporating spatially adjacent information substantially improves prediction accuracy, with diminishing returns beyond spatial extents of approximately 5-7 m. This characteristic scale indicates that ground temperatures are influenced not only by local surface properties, but also by horizontal heat transfer and radiative interactions operating across neighboring microhabitats. The magnitude of spatial effects varied systematically with time of day, microhabitat type, and local environmental characteristics, highlighting context-dependent spatial coupling in microclimate formation. By treating deep learning as a diagnostic tool rather than solely a predictive one, our approach provides a general and transferable method for quantifying spatial dependencies in microclimate models and informing the development of hybrid mechanistic-data-driven approaches that explicitly account for spatial interactions while retaining physical interpretability.



# **Repository Directory**:
## See the  `data` subdirectory for data and metadata. [link](https://github.com/levyofi/Sulami_et_al_Ecology/tree/main/data)

## See the `code` subdirectory for codes. [link](https://github.com/levyofi/Sulami_et_al_Ecology/tree/main/code)

## See the `models` subdirectory for the trained models for each tile size. [link](https://github.com/levyofi/Sulami_et_al_Ecology/tree/main/models)

## See the `predicted_maps` subdirectory for examples of predicted temperature maps and errors. [link](https://github.com/levyofi/Sulami_et_al_Ecology/tree/main/predicted_maps)


