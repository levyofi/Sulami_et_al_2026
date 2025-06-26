from itertools import product
import numpy as np
import torch
import rasterio as rs
import pandas as pd
from torch.utils.data import DataLoader
from dataset_creator import Dataset
import os
import pickle
import matplotlib.pyplot as plt
from models import ResNet
import glob
from tqdm import tqdm
import argparse

class MicroclimateModelAnalysis:

    def __init__(self, maps_folder, models_folder, maps_transform, met_data_transform, map_size=31, device=torch.device('cuda')):
        self.maps_folder = maps_folder
        self.models_folder = models_folder
        self.model = None
        self.maps_transform = maps_transform
        self.met_data_transform = met_data_transform
        self.map_size = map_size
        self.map_types = ['height', 'shade', 'real_solar', 'skyview', 'TGI']
        self.device = device
        self.predicted_maps = {}
        self.predicted_maps_raster_metadata = {}
        self.true_maps = {}
        self.error_maps = {}

    def get_best_model(self):
        pattern = os.path.join(
            self.models_folder,
            f"best_pixel*size_{self.map_size}_*100_epochs.pth"
        )
        model_path = glob.glob(pattern)
        resnet = ResNet(spatial_size=self.map_size).to(self.device)
        model_checkpoint = torch.load(model_path[0], map_location=self.device, weights_only=True)
        resnet.load_state_dict(model_checkpoint)
        self.model = resnet

    def predict_map(self, flight, batch_size=2000, tile_size=64):
        with torch.no_grad():
            # Load the original full image
            input_maps = np.array([
                rs.open(os.path.join(self.maps_folder, flight[:-2], f'{img}_{flight[-1]}.tif')).read(masked=True)[0]
                for img in self.map_types
            ])

            # Load one raster for metadata
            with rs.open(os.path.join(self.maps_folder, flight[:-2], f'{self.map_types[0]}_{flight[-1]}.tif')) as src:
                self.predicted_maps_raster_metadata[flight] = src.meta.copy()

            # Apply maps_transform once to the entire image
            input_tensor = self.maps_transform(torch.Tensor(input_maps)).to(self.device)

            # Load and prepare meteorological data
            met_data = torch.Tensor(
                pd.read_csv(os.path.join(self.maps_folder, flight[:-2], 'meteorological_data.csv')).values[0][1:-3].astype(
                    'float32')
            ).unsqueeze(0).to(self.device)
            trans_met_data = self.met_data_transform(
                met_data.view(met_data.shape[-1], 1, 1)
            ).view(met_data.shape[-1])

            # Prepare patches + metadata
            size = input_maps.shape[-1]
            data = [
                (input_tensor[:, i:i + self.map_size, j:j + self.map_size], trans_met_data)
                for i in range(size - self.map_size)
                for j in range(size - self.map_size)
            ]
            irloader = DataLoader(data, batch_size=batch_size)

            # Make predictions
            all_predictions = torch.cat([
                self.model(img.to(self.device), met.to(self.device))
                for img, met in tqdm(irloader, desc="Predicting patches")
            ])
            # Reshape predictions back to map
            _, H, W = input_tensor.shape
            output_H = H - self.map_size
            output_W = W - self.map_size
            ir_map = all_predictions.view(output_H, output_W)
        self.predicted_maps[flight] = ir_map.cpu()

    def get_and_align_map(self, flight_map, map_name):
        # Read the raster
        raster_path = os.path.join(self.maps_folder, flight_map[:-2], f'{map_name}_{flight_map[-1]}.tif')
        with rs.open(raster_path) as src:
            raster_data = src.read()

        # Mask no data values
        raster_data = np.where(
            (raster_data > 100) | (raster_data < -100),
            np.nan,
            raster_data
        )
        raster_data_map = torch.from_numpy(raster_data)
        # Crop the true map to match the predicted size
        _, H_true, W_true = raster_data_map.shape

        H_pred, W_pred = self.predicted_maps[flight_map].shape

        start_h = (H_true - H_pred) // 2
        end_h = start_h + H_pred

        start_w = (W_true - W_pred) // 2
        end_w = start_w + W_pred

        cropped_true_map = raster_data_map[0, start_h:end_h, start_w:end_w]

        return cropped_true_map

    def get_true_microclimate_map(self, flight_map):
        self.true_maps[flight_map] = self.get_and_align_map(flight_map, map_name="IR")

    def get_shade_map(self, flight_map):
        return self.get_and_align_map(flight_map, map_name="shade")

    def save_predicted_raster(self, flight, error_map = False):
        if (error_map):
            ir_map = self.error_maps[flight].cpu().numpy()
            save_path = os.path.join('predicted_maps', flight[:-2], f'error_size_{self.map_size}_{flight[-1]}.tif')
        else:
            ir_map = self.predicted_maps[flight].cpu().numpy()
            # Add the mean of the true map to shift from centered to actual temperatures
            true_map = self.true_maps[flight].cpu().numpy()
            ir_map = ir_map + np.nanmean(true_map)
            save_path = os.path.join('predicted_maps', flight[:-2], f'predicted_size_{self.map_size}_{flight[-1]}.tif')

        if flight not in self.predicted_maps_raster_metadata:
            # Load one raster for metadata
            with rs.open(os.path.join(self.maps_folder, flight[:-2], f'{self.map_types[0]}_{flight[-1]}.tif')) as src:
                self.predicted_maps_raster_metadata[flight] = src.meta.copy()

        # Adjust metadata: set 1 band, correct height and width
        meta = self.predicted_maps_raster_metadata[flight].copy()

        # --- Calculate the transform correction ---
        # Original dimensions
        original_height = meta["height"]
        original_width = meta["width"]
        predicted_height, predicted_width = ir_map.shape

        # How much was cropped
        crop_top = (original_height - predicted_height) // 2
        crop_left = (original_width - predicted_width) // 2

        # Pixel size (assuming square pixels)
        pixel_size_x = meta["transform"].a
        pixel_size_y = meta["transform"].e  # usually negative because Y is flipped in rasters

        # Adjust the transform
        new_transform = rs.transform.Affine(
            meta["transform"].a,
            meta["transform"].b,
            meta["transform"].c + crop_left * pixel_size_x,
            meta["transform"].d,
            meta["transform"].e,
            meta["transform"].f + crop_top * pixel_size_y
        )

        # Update metadata
        meta.update({
            "count": 1,  # Only one band
            "height": predicted_height,
            "width": predicted_width,
            "transform": new_transform
        })

        # --- Save to file ---
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        with rs.open(save_path, "w", **meta) as dst:
            dst.write(ir_map, 1)  # Write to band 1

    def predict_all_test_maps(self, maps_csv):
        # Generates all IR maps of the test set from a given file
        all_maps = pd.read_csv(maps_csv)
        test_maps = list(all_maps[all_maps['test'] == 1]['Map'])
        for test_map in test_maps:
            for i in range(1, 6):
                map_name = f'{test_map}_{i}'
                print(f"Predicting {map_name}")
                self.predict_map(map_name)

    def get_all_true_test_maps(self, maps_csv):
        # Read all true thermal maps of the test set from a given file
        all_maps = pd.read_csv(maps_csv)
        test_maps = list(all_maps[all_maps['test'] == 1]['Map'])
        for test_map in test_maps:
            for i in range(1, 6):
                map_name = f'{test_map}_{i}'
                print(f"Getting true {map_name}")
                self.get_true_microclimate_map(map_name)

    def calculate_model_error_map(self, flight):
        # Ensure maps exist
        if flight not in self.predicted_maps or self.predicted_maps[flight] is None:
            print(f"No predicted map for flight {flight} was found. Predicting...")
            self.predict_map(flight, batch_size=10000)
        if flight not in self.true_maps or self.true_maps[flight] is None:
            print(f"No true map for flight {flight} was found. Reading...")
            self.get_true_microclimate_map(flight)

        predicted_map = self.predicted_maps[flight]
        true_map = self.true_maps[flight]

        # Shift predicted map so its mean matches true map mean
        shifted_predicted_map = predicted_map + torch.nanmean(true_map)

        # Calculate error map
        self.error_maps[flight] = shifted_predicted_map - true_map

    def calculate_and_save_all_error_maps(self, maps_csv, pkl_filename=None, csv_filename=None):
        """Calculates error maps for all flights and saves them to a pickle file."""
        # List to store error statistics
        error_stats = []
        all_maps = pd.read_csv(maps_csv)
        test_maps = list(all_maps[all_maps['test'] == 1]['Map'])
        for test_map in test_maps:
            for i in range(1, 6):
                map_name = f'{test_map}_{i}'
                print(f"Getting errors for {map_name}")
                self.calculate_model_error_map(map_name)

                # Get the error map
                error_map = self.error_maps[map_name]

                # Calculate metrics
                mean_error = torch.nanmean(error_map).item()
                mean_abs_error = torch.nanmean(torch.abs(error_map)).item()
                mean_sq_error = torch.nanmean(error_map ** 2).item()

                # Save metrics
                error_stats.append({
                    "flight": map_name,
                    "microhabitat": "all",
                    "ME": mean_error,
                    "MAE": mean_abs_error,
                    "MSE": mean_sq_error,
                    "model_size": self.map_size
                })

                #----- Get shade/open errors --- #
                # ---- Load shade raster ----
                shade_tensor = self.get_shade_map(map_name)
                shade_tensor = torch.tensor(shade_tensor, dtype=torch.bool)

                # ---- Calculate statistics for shaded pixels ----
                error_shade = error_map[shade_tensor]
                mean_error_shade = torch.nanmean(error_shade).item()
                mean_abs_error_shade = torch.nanmean(torch.abs(error_shade)).item()
                mean_sq_error_shade = torch.nanmean(error_shade ** 2).item()

                error_stats.append({
                    "flight": map_name,
                    "microhabitat": "shade",
                    "ME": mean_error_shade,
                    "MAE": mean_abs_error_shade,
                    "MSE": mean_sq_error_shade,
                    "model_size": self.map_size
                })

                # ---- Calculate statistics for open pixels ----
                error_open = error_map[~shade_tensor]
                mean_error_open = torch.nanmean(error_open).item()
                mean_abs_error_open = torch.nanmean(torch.abs(error_open)).item()
                mean_sq_error_open = torch.nanmean(error_open ** 2).item()

                error_stats.append({
                    "flight": map_name,
                    "microhabitat": "open",
                    "ME": mean_error_open,
                    "MAE": mean_abs_error_open,
                    "MSE": mean_sq_error_open,
                    "model_size": self.map_size
                })

        # Define filenames
        if pkl_filename is None:
            pkl_filename = f'error_maps_model_size_{self.map_size}.pkl'
        if csv_filename is None:
            csv_filename = f'error_maps_model_size_{self.map_size}.csv'

        # Save the error maps
        with open(pkl_filename, 'wb') as f:
            pickle.dump(self.error_maps, f)

        # Save the statistics as a CSV
        df_stats = pd.DataFrame(error_stats)
        df_stats.to_csv(csv_filename, index=False)

        print(f"Saved error maps of model size {self.map_size} for {len(self.error_maps)} flights to '{pkl_filename}' and '{csv_filename}'.")

    def save_predicted_maps_as_rasters(self):
        for flight in self.predicted_maps.keys():
            self.save_predicted_raster(flight)

    def save_error_maps_as_rasters(self):
        for flight in self.error_maps.keys():
            self.save_predicted_raster(flight, error_map=True)

    def save_maps(self):
        # Saves the IR maps in a pickle file
        with open(f'predicted_maps_model_size_{self.map_size}.pkl', 'wb') as maps:
            pickle.dump(self.predicted_maps, maps)
        with open(f'true_maps_model_size_{self.map_size}.pkl', 'wb') as maps:
            pickle.dump(self.true_maps, maps)

    def load_maps(self):
        """Loads predicted and true maps from pickle files, with warning if files are missing."""
        pred_maps_file = f'predicted_maps_model_size_{self.map_size}.pkl'
        true_maps_file = f'true_maps_model_size_{self.map_size}.pkl'

        if not os.path.exists(pred_maps_file):
            print(f"Warning: Predicted maps file '{pred_maps_file}' not found. Skipping loading predicted maps.")
            self.predicted_maps = {}
        else:
            with open(pred_maps_file, 'rb') as pred_file:
                self.predicted_maps = pickle.load(pred_file)

        if not os.path.exists(true_maps_file):
            print(f"Warning: True maps file '{true_maps_file}' not found. Skipping loading true maps.")
            self.true_maps = {}
        else:
            with open(true_maps_file, 'rb') as true_file:
                self.true_maps = pickle.load(true_file)

spatial_sizes = [5, 9, 15, 21, 31, 47, 63, 81]

def get_device(preferred="cuda"):
    if preferred == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device

def main():
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
    irg = MicroclimateModelAnalysis(maps_folder, models_folder, trainset.maps_transform, trainset.met_data_transform, map_size = spatial_size, device=device)
    irg.get_best_model()

    # get the maps for a certain flight, calculate error and plot the map
    flight = 'Zeelim_23.9.19_0950_1'
    irg.load_maps()
    predicted_map_cpu = irg.predicted_maps[flight].cpu().numpy()
    irg.get_true_microclimate_map(flight)
    true_map_cpu = irg.true_maps[flight].cpu().numpy()
    predicted_map_cpu = predicted_map_cpu+np.nanmean(true_map_cpu)
    error_map = predicted_map_cpu - true_map_cpu
    error = np.nanmean(error_map)
    print(error)
    irg.save_predicted_raster(flight)

    plt.imshow(error_map, cmap='inferno')  # or any suitable colormap
    plt.colorbar(label='Temperature Prediction')
    plt.title('Predicted Ground Temperature Map')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')

    # Show plot and wait until the user closes the window
    plt.show(block=False)
    input("Press Enter to continue and close the plot...")
    plt.close()


if __name__ == '__main__':
    main()
