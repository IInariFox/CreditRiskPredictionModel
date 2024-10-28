import kaggle
import os

# Ensure the 'data' directory exists
os.makedirs('data', exist_ok=True)

# Kaggle dataset identifier
dataset = 'wordsforthewise/lending-club'

# Download and unzip the dataset into the 'data' directory
kaggle.api.dataset_download_files(dataset, path='data/', unzip=True)

print("Dataset downloaded and unzipped in the 'data/' directory.")