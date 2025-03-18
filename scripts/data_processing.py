import os
import sys
import pandas as pd
import pickle

# Add the package to path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from march_madness.config import (STARTING_SEASON, CURRENT_SEASON, 
                                 TRAINING_SEASONS, VALIDATION_SEASON, 
                                 PREDICTION_SEASONS)
from march_madness.data.loaders import load_mens_data, load_womens_data
from march_madness.utils.data_processors import prepare_modeling_data

def main():
    print("Processing NCAA Basketball Tournament Data")
    
    # Create cache directory
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    # Define the data directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, 'data')
    
    # Process men's data
    print("\nProcessing men's data...")
    mens_data = load_mens_data(STARTING_SEASON, data_dir=data_dir)
    mens_modeling_data = prepare_modeling_data(
        mens_data, "men's", STARTING_SEASON, CURRENT_SEASON, 
        TRAINING_SEASONS + [VALIDATION_SEASON] + PREDICTION_SEASONS,
        cache_dir=cache_dir
    )
    print("Men's data processing complete and cached")
    
    # Process women's data
    print("\nProcessing women's data...")
    womens_data = load_womens_data(STARTING_SEASON, data_dir=data_dir)
    womens_modeling_data = prepare_modeling_data(
        womens_data, "women's", STARTING_SEASON, CURRENT_SEASON, 
        TRAINING_SEASONS + [VALIDATION_SEASON] + PREDICTION_SEASONS,
        cache_dir=cache_dir
    )
    print("Women's data processing complete and cached")
    
    print("\nAll data processing complete! The processed data is cached in the 'cache' directory.")

if __name__ == "__main__":
    main()