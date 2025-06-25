import numpy as np
import pandas as pd
import rasterio as rs
import os
import pickle
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class Dataset(Dataset):

    def __init__(self, wd, is_train=True, is_validation=False, transform=True):
        self.wd = wd
        self.map_types = ['height', 'shade', 'real_solar', 'skyview', 'TGI']
        self.is_train = is_train
        self.is_validation = is_validation
        self.train_test_split()
        self.transform = transform
        self.data = []

    def train_test_split(self):
        # Splits the data into train and test sets with respect to a given file
        all_maps = pd.read_csv('data/desert_maps.csv')
        train_or_val = 'validation' if self.is_validation else 'train'
        self.train_list = list(all_maps[all_maps[train_or_val] == 1]['Map'])
        self.test_list = list(all_maps[all_maps['test'] == 1]['Map'])
        self.all_maps = self.train_list + self.test_list

    def load_maps(self):
        # Loads the maps
        map_list = self.train_list if self.is_train else self.test_list
        for flight in map_list:
            met_data = torch.Tensor(
                pd.read_csv(os.path.join(self.wd, flight, 'meteorological_data.csv')).values[0][1:-3].astype('float32'))
            for i in range(1, 6):
                maps = torch.Tensor(np.array(
                    [rs.open(os.path.join(self.wd, flight, f'{map_type}_{i}.tif')).read(masked=True)[0] for map_type in
                     self.map_types]))
                labels = torch.Tensor(np.array(rs.open(os.path.join(self.wd, flight, f'IR_{i}.tif')).read(masked=True)))
                labels = self.fix_extreme_values(labels)
                self.data.append((maps, met_data, labels))
        if self.transform:
            self.compute_maps_transform()
            self.compute_met_data_transform()

    def fix_extreme_values(self, image):
        # Fixes extreme values by replacing them with the mean
        image[abs(image) > 1e6] = torch.mean(image[abs(image) < 1e6])
        return image

    def compute_maps_transform(self):
        # Computes the maps normalization transform
        images = torch.stack([item[0] for item in self.data], dim=3)
        images_mean = images.view(5, -1).mean(dim=1)
        images_std = images.view(5, -1).std(dim=1)
        self.maps_transform = transforms.Normalize(images_mean, images_std)

    def compute_met_data_transform(self):
        # Computes the meteorological data normalization transform
        met_data = torch.stack([item[1] for item in self.data], dim=1)
        self.met_data_transform = transforms.Normalize(met_data.mean(dim=1), met_data.std(dim=1))

    def save_data(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_data(filename):
        with open(filename, 'rb') as f:
            obj = pickle.load(f)
        return obj

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        maps, met_data, labels = self.data[index]
        if self.transform:
            maps = self.maps_transform(maps)
            met_data = self.met_data_transform(met_data.view(met_data.shape[0], 1, 1)).view(met_data.shape[0])
            labels = labels - labels.mean()
        return maps, met_data, labels


    # def extract_validation(self):
    #     # Extracts every 5 maps from a dataset to form a validation set
    #     val_indexes = range(4, len(self), 5)
    #     valset = [self[i] for i in val_indexes]
    #     trainset = [self[i] for i in range(len(self)) if i not in val_indexes]
    #     return trainset, valset

    def extract_validation(self):
        val_indexes = set(range(4, len(self), 5))

        # Create empty Dataset instances without calling __init__
        trainset = Dataset.__new__(Dataset)
        valset = Dataset.__new__(Dataset)

        # Manually assign required attributes
        for dataset in [trainset, valset]:
            dataset.wd = self.wd
            dataset.map_types = self.map_types
            dataset.is_train = self.is_train
            dataset.is_validation = self.is_validation
            dataset.transform = self.transform

        # Split data
        trainset.data = [self.data[i] for i in range(len(self)) if i not in val_indexes]
        valset.data = [self.data[i] for i in val_indexes]

        # Compute transforms on training data ONLY
        if self.transform:
            trainset.compute_maps_transform()
            trainset.compute_met_data_transform()

            # Use training transforms for validation set
            valset.maps_transform = trainset.maps_transform
            valset.met_data_transform = trainset.met_data_transform

        return trainset, valset

    # add a split function
    def split_maps_with_overlap(self, chunk_size=31, overlap=21, to_pixel=True):
        # Splits images of a dataset into chunks of a given size with a given overlap
        split_maps = []
        height, width = self[0][0].shape[1:]
        for data, met_data, labels in self:
            data_patches = []
            labels_patches = []
            for i in range(0, height - chunk_size + 1, chunk_size - overlap):
                for j in range(0, width - chunk_size + 1, chunk_size - overlap):
                    data_patches.append(data[:, i:i + chunk_size, j:j + chunk_size])
                    labels_patches.append(labels[:, i:i + chunk_size, j:j + chunk_size])
            if to_pixel:
                center = int(chunk_size / 2)
                labels_patches = list(map(lambda image: image[0][center][center].item(), labels_patches))
            split_maps += [(data_patch, met_data, labels_patch) for data_patch, labels_patch in
                           zip(data_patches, labels_patches)]
        return split_maps
    # add map normalization for the thermal maps based on a number of pixels

def main():
    WD = '/big_data/idan/complete_subimages_cropped'

    # create the train set - 25 flights
    trainset = Dataset(WD)
    trainset.load_maps()
    trainset_path = '/home/ofir/Dropbox/pycharm_projects/Sulami_et_al_Ecology/data/trainset_ofir.pkl'
    trainset.save_data(trainset_path)

    # create the test set - 7 flights
    testset = Dataset(WD, is_train=False, is_validation=True)
    testset.load_maps()
    testset_path = '/home/ofir/Dropbox/pycharm_projects/Sulami_et_al_Ecology/data/testset_ofir.pkl'
    testset.save_data(testset_path)

    print("Dataset Creation Ended")


if __name__ == '__main__':
    main()
