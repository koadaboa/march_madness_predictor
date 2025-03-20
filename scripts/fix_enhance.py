#!/usr/bin/env python
"""
Script to enhance NCAA Basketball Tournament prediction models with better
data loading to fix the tournament data issue.
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the package to path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from march_madness.config import (TRAINING_SEASONS, VALIDATION_SEASON, PREDICTION_SEASONS)

def main():
    """Main function to enhance models and generate improved predictions"""
    start_time = datetime.now()
    logger.info(f"Starting model enhancement at {start_time}")
    
    # Setup directories
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, 'data')
    cache_dir = os.path.join(project_root, 'cache')
    models_dir = os.path.join(project_root, 'models')
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # Define prediction seasons
    prediction_seasons = PREDICTION_SEASONS
    logger.info(f"Will generate predictions for seasons: {prediction_seasons}")
    
    # Load cached modeling data
    logger.info("Loading cached modeling data...")
    
    # Men's modeling data
    mens_cache_file = os.path.join(cache_dir, "men's_modeling_data.pkl")
    if os.path.exists(mens_cache_file):
        with open(mens_cache_file, 'rb') as f:
            mens_modeling_data = pickle.load(f)
        logger.info(f"Loaded men's modeling data from {mens_cache_file}")
    else:
        logger.error(f"Men's modeling data not found at {mens_cache_file}")
        return

    # Women's modeling data
    womens_cache_file = os.path.join(cache_dir, "women's_modeling_data.pkl")
    if os.path.exists(womens_cache_file):
        with open(womens_cache_file, 'rb') as f:
            womens_modeling_data = pickle.load(f)
        logger.info(f"Loaded women's modeling data from {womens_cache_file}")
    else:
        logger.error(f"Women's modeling data not found at {womens_cache_file}")
        return
    
    # Load original models
    logger.info("Loading original models...")
    
    # Men's model
    mens_model_file = os.path.join(models_dir, 'mens_model.pkl')
    mens_metadata_file = os.path.join(models_dir, 'mens_metadata.pkl')
    
    if os.path.exists(mens_model_file) and os.path.exists(mens_metadata_file):
        with open(mens_model_file, 'rb') as f:
            mens_model = pickle.load(f)
        with open(mens_metadata_file, 'rb') as f:
            mens_metadata = pickle.load(f)
        logger.info("Loaded original men's model")
    else:
        logger.error(f"Original men's model files not found")
        return
    
    # Women's model
    womens_model_file = os.path.join(models_dir, 'womens_model.pkl')
    womens_metadata_file = os.path.join(models_dir, 'womens_metadata.pkl')
    
    if os.path.exists(womens_model_file) and os.path.exists(womens_metadata_file):
        with open(womens_model_file, 'rb') as f:
            womens_model = pickle.load(f)
        with open(womens_metadata_file, 'rb') as f:
            womens_metadata = pickle.load(f)
        logger.info("Loaded original women's model")
    else:
        logger.error(f"Original women's model files not found")
        return
    
    # Load tournament data directly from CSV files
    logger.info("Loading tournament data directly from CSV files...")
    
    # Men's tournament data
    mens_tourney_file = os.path.join(data_dir, 'MNCAATourneyDetailedResults.csv')
    if os.path.exists(mens_tourney_file):
        mens_tourney = pd.read_csv(mens_tourney_file)
        logger.info(f"Loaded {len(mens_tourney)} men's tournament games")
    else:
        logger.error(f"Men's tournament file not found: {mens_tourney_file}")
        return
        
    # Women's tournament data
    womens_tourney_file = os.path.join(data_dir, 'WNCAATourneyDetailedResults.csv')
    if os.path.exists(womens_tourney_file):
        womens_tourney = pd.read_csv(womens_tourney_file)
        logger.info(f"Loaded {len(womens_tourney)} women's tournament games")
    else:
        logger.error(f"Women's tournament file not found: {womens_tourney_file}")
        return
    
    # Process men's tournament data
    logger.info("Processing men's tournament data...")
    train_seasons = [s for s in TRAINING_SEASONS if s != 2020] + [VALIDATION_SEASON]
    mens_train_tourney = mens_tourney[mens_tourney['Season'].isin(train_seasons)]
    logger.info(f"Using men's training seasons: {train_seasons}")
    logger.info(f"Found {len(mens_train_tourney)} men's tournament games for training")
    
    # Update cached modeling data with tournament data
    mens_modeling_data['tourney_data'] = mens_tourney
    
    # Process women's tournament data
    logger.info("Processing women's tournament data...")
    womens_train_tourney = womens_tourney[womens_tourney['Season'].isin(train_seasons)]
    logger.info(f"Found {len(womens_train_tourney)} women's tournament games for training")
    
    # Update cached modeling data with tournament data
    womens_modeling_data['tourney_data'] = womens_tourney
    
    # Save the updated cache files
    logger.info("Saving updated cache files with tournament data...")
    with open(mens_cache_file, 'wb') as f:
        pickle.dump(mens_modeling_data, f)
    
    with open(womens_cache_file, 'wb') as f:
        pickle.dump(womens_modeling_data, f)
    
    logger.info("Cache files updated with tournament data")
    logger.info("Now you can run enhance_models.py again to create enhanced models")
    
    # Print instructions
    print("\n" + "="*60)
    print("INSTRUCTIONS:")
    print("1. We've updated your cache files with the tournament data")
    print("2. Now run the original enhance_models.py script:")
    print("   python scripts/enhance_models.py")
    print("="*60)

if __name__ == "__main__":
    main()