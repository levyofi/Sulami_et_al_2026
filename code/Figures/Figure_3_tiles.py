import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Load the npy file
ground_temps_npy = np.load('Zeelim_23.9.19_0800_2_IR.npy')

# Define the base crop size and additional sizes
base_crop_size = 63
crop_sizes = [63, 47, 31, 21, 15, 9]
pixel_size_cm = 15  # Each pixel represents 15 cm

# Function to find a 61x61 crop without NaN values
def find_valid_crop(data, crop_size):
    rows, cols = data.shape
    for i in range(rows - crop_size + 1):
        for j in range(cols - crop_size + 1):
            crop = data[i:i + crop_size, j:j + crop_size]
            if not np.isnan(crop).any():
                return crop
    return None

# Get a 61x61 valid crop with no NaN values
initial_crop = find_valid_crop(ground_temps_npy, base_crop_size)

if initial_crop is not None:
    # Center of the 61x61 crop
    center_x, center_y = base_crop_size // 2, base_crop_size // 2

    # Set up a 3x2 figure with subplots and a colorbar on the right
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    min_temp, max_temp = np.nanmin(initial_crop), np.nanmax(initial_crop)  # Color scale based on data

    # Flatten axes for easy iteration
    axes = axes.flatten()

    # Plot each crop on a separate subplot
    for ax, size in zip(axes, crop_sizes):
        # Calculate the starting and ending indices for the new crop around the center
        start_x = center_x - size // 2
        end_x = center_x + size // 2 + 1
        start_y = center_y - size // 2
        end_y = center_y + size // 2 + 1
        crop = initial_crop[start_x:end_x, start_y:end_y]

        # Calculate the real-world size in centimeters
        real_size_cm = size * pixel_size_cm

        # Display the crop with 'jet' colormap and black grid lines
        img = ax.imshow(crop, cmap='jet', interpolation='nearest', vmin=min_temp, vmax=max_temp)

        # Add black grid lines
        for x in range(size + 1):
            ax.axhline(x - 0.5, color='black', linewidth=0.5)
            ax.axvline(x - 0.5, color='black', linewidth=0.5)

        # Set subplot title with real-world size in cm
        ax.set_title(f'{size}×{size} Pixels ({real_size_cm} cm²)', fontsize=20)
        ax.axis('off')  # Turn off axes for clarity

    # Remove any unused subplots
    for ax in axes[len(crop_sizes):]:
        ax.axis('off')

    # Add a single colorbar on the right of the figure with increased width
    cbar = fig.colorbar(img, ax=axes, orientation='vertical', fraction=0.07, pad=0.1)
    cbar.set_label('Ground Temperature (°C)', fontsize=20)
    cbar.ax.tick_params(labelsize=18)  # Increase font size of temperature labels

    # Set main title with larger font size
    #plt.suptitle("Tiles of Varying Sizes", fontsize=16)
    #plt.suptitle("", fontsize=16)

    # Save the figure to a file
    plt.savefig("Figure_3_tiles.png", dpi=300)

    # Show the plot
    plt.show()

else:
    print("No 61x61 crop without NaN values found in the data.")
