import os
import sys
import pandas as pd
import numpy as np
import pickle

# Add the package to path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from march_madness.config import STARTING_SEASON
from march_madness.data.loaders import load_mens_data, load_womens_data, extract_seed_number
from march_madness.utils.data_processors import prepare_modeling_data
from march_madness.utils.helpers import train_and_predict_model
from march_madness.models.prediction import combine_predictions

def main():
    print("Generating 2025 NCAA Basketball Tournament Predictions")
    
    # Define directories
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, 'data')
    models_dir = os.path.join(project_root, 'models')
    
    # Set 2025 as the prediction season
    PREDICTION_SEASON = 2025
    
    # Load the 2025 seed data
    print("Loading 2025 seed data...")
    try:
        # Try to load the 2025 seed files
        mens_2025_seeds = pd.read_csv(os.path.join(data_dir, 'MNCAATourneySeeds2025.csv'))
        womens_2025_seeds = pd.read_csv(os.path.join(data_dir, 'WNCAATourneySeeds2025.csv'))
        
        # Ensure we have the expected columns
        if 'Season' not in mens_2025_seeds.columns:
            mens_2025_seeds['Season'] = PREDICTION_SEASON
        if 'Season' not in womens_2025_seeds.columns:
            womens_2025_seeds['Season'] = PREDICTION_SEASON
            
        print(f"Found {len(mens_2025_seeds)} men's teams and {len(womens_2025_seeds)} women's teams for 2025")
    except FileNotFoundError:
        print("2025 seed files not found. Falling back to loading all team data...")
        # Load team lists from regular data files
        mens_data = load_mens_data(STARTING_SEASON, data_dir=data_dir)
        womens_data = load_womens_data(STARTING_SEASON, data_dir=data_dir)
        
        # Get team IDs from MTeams.csv and WTeams.csv instead of seed files
        if 'df_teams' in mens_data:
            mens_teams = mens_data['df_teams'].copy()
            mens_teams = mens_teams[mens_teams['LastD1Season'] >= 2024]  # Only active teams
            mens_2025_teams = mens_teams['TeamID'].unique()
        else:
            print("ERROR: Could not find men's teams data")
            return
            
        if 'df_teams' in womens_data:
            womens_teams = womens_data['df_teams'].copy()
            womens_2025_teams = womens_teams['TeamID'].unique()
        else:
            print("ERROR: Could not find women's teams data")
            return
            
        print(f"Found {len(mens_2025_teams)} men's teams and {len(womens_2025_teams)} women's teams")
        
        # Create placeholder seed files
        mens_2025_seeds = pd.DataFrame({
            'Season': [PREDICTION_SEASON] * len(mens_2025_teams),
            'TeamID': mens_2025_teams,
            'Seed': ['X01'] * len(mens_2025_teams)  # Placeholder seeds
        })
        
        womens_2025_seeds = pd.DataFrame({
            'Season': [PREDICTION_SEASON] * len(womens_2025_teams),
            'TeamID': womens_2025_teams,
            'Seed': ['X01'] * len(womens_2025_teams)  # Placeholder seeds
        })
    
    # Load the trained models and metadata
    print("Loading trained models...")
    with open(os.path.join(models_dir, 'mens_model.pkl'), 'rb') as f:
        mens_model = pickle.load(f)
    
    with open(os.path.join(models_dir, 'mens_metadata.pkl'), 'rb') as f:
        mens_metadata = pickle.load(f)
    
    with open(os.path.join(models_dir, 'womens_model.pkl'), 'rb') as f:
        womens_model = pickle.load(f)
    
    with open(os.path.join(models_dir, 'womens_metadata.pkl'), 'rb') as f:
        womens_metadata = pickle.load(f)
    
    # Load the complete data to use for feature engineering
    print("Loading historical data for feature engineering...")
    mens_data = load_mens_data(STARTING_SEASON, data_dir=data_dir)
    womens_data = load_womens_data(STARTING_SEASON, data_dir=data_dir)
    
    # Add 2025 seed data to the seed dataframe
    if 'df_seed' in mens_data:
        # Make a copy of the existing seed data and append 2025 seeds
        mens_seeds = mens_data['df_seed'].copy()
        mens_seeds = pd.concat([mens_seeds, mens_2025_seeds], ignore_index=True)
        mens_data['df_seed'] = mens_seeds
    else:
        mens_data['df_seed'] = mens_2025_seeds
    
    if 'df_seed' in womens_data:
        # Make a copy of the existing seed data and append 2025 seeds
        womens_seeds = womens_data['df_seed'].copy()
        womens_seeds = pd.concat([womens_seeds, womens_2025_seeds], ignore_index=True)
        womens_data['df_seed'] = womens_seeds
    else:
        womens_data['df_seed'] = womens_2025_seeds
    
    # Process the data for 2025 predictions
    print(f"Preparing data for 2025 predictions...")
    
    mens_2025_data = prepare_modeling_data(
        mens_data, "men's", STARTING_SEASON, PREDICTION_SEASON, [PREDICTION_SEASON]
    )
    
    womens_2025_data = prepare_modeling_data(
        womens_data, "women's", STARTING_SEASON, PREDICTION_SEASON, [PREDICTION_SEASON]
    )
    
    # Generate predictions
    print("Generating Men's 2025 predictions...")
    mens_predictions = train_and_predict_model(
        mens_2025_data, "men's", [], None, [PREDICTION_SEASON],
        model=mens_model,
        feature_cols=mens_metadata['feature_cols'],
        scaler=mens_metadata['scaler'],
        dropped_features=mens_metadata['dropped_features']
    )
    
    print("Generating Women's 2025 predictions...")
    womens_predictions = train_and_predict_model(
        womens_2025_data, "women's", [], None, [PREDICTION_SEASON],
        model=womens_model,
        feature_cols=womens_metadata['feature_cols'],
        scaler=womens_metadata['scaler'],
        dropped_features=womens_metadata['dropped_features']
    )
    
    # Combine predictions
    print("Combining predictions...")
    combined_submission = combine_predictions(mens_predictions, womens_predictions)
    
    # Save predictions
    submission_path = os.path.join(project_root, "submission_2025.csv")
    combined_submission.to_csv(submission_path, index=False)
    print(f"Predictions generated and saved to '{submission_path}'")
    print(f"Total predictions: {len(combined_submission)} matchups")

if __name__ == "__main__":
    main()