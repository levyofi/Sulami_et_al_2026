import os
import argparse
import glob
from MicroclimateModelAnalysis import MicroclimateModelAnalysis
from MicroclimateModelAnalysis import get_device
from dataset_creator import Dataset

# -------------- #
# Argument Parser #
# -------------- #
parser = argparse.ArgumentParser(description="Get model predictions and errors by spatial size.")
parser.add_argument('--size', type=int, required=True, help='Spatial size to filter the models')
args = parser.parse_args()

#maps_folder = '/big_data/idan/complete_subimages_cropped' #location on old server
#maps_folder = "/data/idan/complete_subimages_cropped" #location on new server

maps_folder = os.path.expanduser("data/submaps_cropped")
models_folder = os.path.expanduser('models')
# load the training data
trainset_path = os.path.expanduser('data/trainset_ofir.pkl')
# check the range of temperatures in each train map
trainset = Dataset.load_data(trainset_path)
print("Dataset Loaded")
# get the data of the desert maps
maps_csv = os.path.expanduser('data/desert_maps.csv')

#Testing the code
spatial_size = args.size
# load the model
device = get_device(preferred="cuda")  # Try "cuda", fallback to "cpu" if not available
print(f"Using device: {device}")
# Set the analysis object
microclimate_analysis = MicroclimateModelAnalysis(maps_folder, models_folder, trainset.maps_transform, trainset.met_data_transform, map_size = spatial_size, device=device)
# Get the best model for this spatial size
microclimate_analysis.get_best_model()
# Load previously predicted and true maps if already saved from pkl files
microclimate_analysis.load_maps()
# Calculate the errors - will calculate any missing predicted and true maps
microclimate_analysis.calculate_and_save_all_error_maps(maps_csv)
# Save all predicted and true maps to pkl files
microclimate_analysis.save_maps()
# Save all predicted and error maps as raster maps
microclimate_analysis.save_predicted_maps_as_rasters()
microclimate_analysis.save_error_maps_as_rasters()

