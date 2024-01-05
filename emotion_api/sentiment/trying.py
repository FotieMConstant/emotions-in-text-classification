from pathlib import Path
import os

# Get the directory of the script
script_dir = Path(__file__).resolve().parent

# Define the relative path to the model
model_relative_path = '../../dbert_model.h5'

# Combine the script directory with the relative path to get the absolute path
model_save_path = script_dir / model_relative_path

print(model_save_path)
