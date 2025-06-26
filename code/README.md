# Code Directory

This directory contains all scripts and modules used for data processing, modeling, and analysis in the Sulami et al. Ecology project. The workflow is divided into several stages, from data creation to model training/testing and statistical analysis.

## Main Files

- **stage_1_dataset_creator.py**: Creates the dataset from the raw data files located in the `../data` folder. Generates intermediate files for further processing.
- **stage_2_explore_dataset.py**: Performs exploratory data analysis on the dataset created in stage 1.
- **stage_3_run_parallel_models_trainer.py**: Trains multiple models in parallel using the processed datasets. Outputs model files to the `../models` directory.
- **stage_4_run_parallel_model_testers.py**: Tests the trained models in parallel. Generates prediction results that are saved in the `../predicted_maps` directory.
- **model_trainer.py**: Contains code for training individual models.
- **model_tester.py**: Contains code for testing individual models and evaluating performance.
- **models.py**: Defines the model architectures and utility functions.
- **model_comparison.csv**: Stores results of model comparisons.
- **MicroclimateModelAnalysis.py**: Main script for analyzing microclimate models.

## Subdirectories

- **Figures/**: Scripts and RData files for figure generation and visualization. See `Figures/README.md` for details.
- **statistical_analysis/**: Contains R scripts and data for statistical analysis of model outputs. See `statistical_analysis/README.md` for more information.

## Workflow Overview

1. **Data Creation**: Use `stage_1_dataset_creator.py` to process raw data from `../data`.
2. **Exploration**: Analyze the created dataset with `stage_2_explore_dataset.py`.
3. **Model Training**: Train models using `stage_3_run_parallel_models_trainer.py` and supporting scripts.
4. **Model Testing**: Evaluate models with `stage_4_run_parallel_model_testers.py`.
5. **Statistical Analysis and Figures**: Move to the respective subfolders for further analysis and figure generation.

## File Dependencies

- Scripts in this folder mainly utilize and generate data in the `../data`, `../models`, and `../predicted_maps` directories.
- Outputs from model testing are further analyzed in `statistical_analysis` and visualized in `Figures`.

See each subfolder's README for more details.
