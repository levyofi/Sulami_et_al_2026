import numpy as np
import pandas as pd
from datetime import datetime

from dataset_creator import Dataset
from models import ResNet
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim

import wandb

from sklearn.model_selection import train_test_split
import os

from MicroclimateModelAnalysis import get_device

class TrainModel:

    def __init__(self, model, data_loader, num_epochs, spatial_size, batch_size, optimizer='adam', lr=0.0001, momentum=0.9,
                 criterion=nn.MSELoss(), val_loader=None, early_stop=False, extended_name=None, device='cuda'):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.data_loader = data_loader
        self.epochs = num_epochs
        self.optimizer = self.get_optimizer(optimizer, lr, momentum)
        self.criterion = criterion
        self.model_name = self.get_model_name(model.model_name, optimizer, extended_name, num_epochs)
        self.val_loader = val_loader
        self.early_stop = early_stop
        self.train_loss = []
        self.batch_size = batch_size
        self.spatial_size = spatial_size

    def get_model_name(self, model_name, optimizer, extended_name, epochs):
        return f'{model_name}_{extended_name}_{epochs}_epochs'

    def get_optimizer(self, optimizer, lr, momentum):
        if optimizer == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer == 'sgd':
            return optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)


    def train(self, verbose_num_batches=1000, save_interval=1, save_title='', save_loss=False):
        early_stopped = False
        best_val_loss = np.inf
        best_epoch_loss = 0
        epochs_no_improve = 0
        if self.early_stop:
            patience, min_delta = self.early_stop  # assuming early_stop=(patience, min_delta)
        else:
            patience = 5
            min_delta = 0.001

        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0

            # iterate through batches
            for i, data in enumerate(self.data_loader):
                inputs, met_data, labels = (chunk.to(self.device) for chunk in data)
                labels = labels.float()
                self.optimizer.zero_grad()
                outputs = self.model(inputs, met_data)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

                if (i + 1) % verbose_num_batches == 0:
                    print(f'Epoch [{epoch + 1}/{self.epochs}], Batch [{i + 1}]')

            avg_loss = running_loss / len(self.data_loader)
            print(f'Epoch [{epoch + 1}/{self.epochs}] - Avg Loss: {avg_loss:.4f}')

            # Compute validation loss at end of epoch
            if self.val_loader is not None:
                val_loss = self.get_validation_loss()
                print(f'Epoch [{epoch + 1}] Validation Loss: {val_loss:.4f}')
                wandb.log({'epoch': epoch + 1, 'val_loss': val_loss, 'train_loss': avg_loss})

                # Early stopping check
                if val_loss < best_val_loss - min_delta:
                    best_val_loss = val_loss
                    best_epoch_loss = epoch + 1
                    epochs_no_improve = 0
                    # optionally, save best model checkpoint here
                    self.save_model(model_name=f"best_" + self.model_name)
                else:
                    if self.early_stop:
                        epochs_no_improve += 1
                        if epochs_no_improve >= patience:
                            early_stopped = True
                            print(
                                f"Early stopping triggered after {epoch + 1} epochs (no improvement for {patience} epochs).")
                            break

            # Save loss history
            self.train_loss.append((epoch + 1, avg_loss))
            if save_loss:
                pd.DataFrame(self.train_loss, columns=['Epochs', 'Train loss']).to_csv(f'{self.model_name}.csv',
                                                                                       index=False)

            # Model checkpointing at intervals
            if (epoch + 1) % save_interval == 0:
                self.save_model(epochs=epoch + 1)

        self.log_summary_to_csv(
            best_val_loss=best_val_loss,
            best_epoch_loss=best_epoch_loss,
            early_stopped=early_stopped
        )

    def get_validation_loss(self):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in self.val_loader:
                inputs, met_data, labels = list(map(lambda chunk: chunk.to(self.device), data))
                labels = labels.float()
                outputs = self.model(inputs, met_data)
                val_loss += self.criterion(outputs, labels).item()
        val_loss /= len(self.val_loader)
        self.model.train()
        return val_loss

    def early_stopping(self, val_loss, best_val_loss, epochs_no_improve):
        min_delta = self.early_stop[1]
        if val_loss + min_delta < best_val_loss:
            return val_loss, 0
        else:
            return best_val_loss, epochs_no_improve + 1

    def save_model(self, model_name="", epochs=None):
        epochs = epochs if epochs else self.epochs
        if model_name=="":
            model_name = '_'.join(self.model_name.split('_')[:-2]) + f'_{epochs}_epochs'
        torch.save(self.model.state_dict(), f'/data/microclimate_training_data/models/{model_name}.pth')

    def log_summary_to_csv(self, best_val_loss, best_epoch_loss, early_stopped):
        import csv
        from datetime import datetime

        summary_file = "model_comparison.csv"
        header = [
            'Model',
            'Epochs',
            'Batch Size',
            'Spatial Size',
            'Best Val Loss',
            'Best Epoch',
            'End Time',
            'Early Stopped'
        ]

        end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        row = [
            self.model_name,
            self.epochs,
            self.batch_size,
            self.spatial_size,
            round(best_val_loss, 5) if best_val_loss is not None else None,
            best_epoch_loss,
            end_time,
            "Yes" if early_stopped else "No"
        ]

        # Create the CSV with header if it doesn't exist
        if not os.path.exists(summary_file):
            with open(summary_file, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)

        # Append the result
        with open(summary_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)




def compute_batch_size(dataset, min_batches=10000):
    dataset_size = len(dataset)
    if dataset_size < min_batches:
        return 1
    return max(1, dataset_size // min_batches)


def main():
    import argparse

    device = get_device(preferred="cuda")  # Try "cuda", fallback to "cpu" if not available
    print(f"Using device: {device}")

    #get the size of the spatial information for training
    parser = argparse.ArgumentParser(description="Arguments for training model")
    parser.add_argument('--size', type=int, help='Size of spatial information')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    args = parser.parse_args()
    spatial_size = args.size
    batch_size = args.batch_size

    wandb.login()

    epochs = 100
    lr = 0.0001

    run = wandb.init(
        project=f"microclimate-{spatial_size}-{batch_size}",  # Specify your project
        config={  # Track hyperparameters and metadata
            "learning_rate": lr,
            "epochs": epochs,
        },
    )
    # load the training data
    trainset_path = os.path.expanduser('~/Dropbox/pycharm_projects/Sulami_et_al_Ecology/data/trainset_ofir.pkl')

    # check the range of temperatures in each train map
    trainset = Dataset.load_data(trainset_path)
    print("Dataset Loaded")

    print("Creating split dataset...")
    split_dataset = trainset.split_maps_with_overlap(
        chunk_size=spatial_size,
        overlap=int(spatial_size - spatial_size * 3 / 4),
        to_pixel=True
    )
    # Now, split at the patch level
    train_data, val_data = train_test_split(
        split_dataset, test_size=0.2, random_state=42, shuffle=True
    )

    trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True)
    print("Batches per epoch:", len(trainloader))

    valloader = DataLoader(val_data, batch_size=batch_size, shuffle=True, pin_memory=True)

    model = TrainModel(ResNet(spatial_size=spatial_size), trainloader, val_loader=valloader, num_epochs=epochs, lr = lr, early_stop=False, extended_name=f'size_{args.size}_batches_{batch_size}', spatial_size=spatial_size, batch_size=batch_size, device=device)
    model.train(save_interval=1, save_loss=True)

if __name__ == '__main__':
    main()
