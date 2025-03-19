#!/usr/bin/env python
"""
Generate NCAA Basketball Tournament Predictions
With real-world mode for 2025 predictions using comprehensive cached data
"""

import os
import pandas as pd
import numpy as np
import pickle
import sys
import time
import argparse
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the package to path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from march_madness.config import (STARTING_SEASON, CURRENT_SEASON, 
                                 TRAINING_SEASONS, VALIDATION_SEASON, 
                                 PREDICTION_SEASONS)
from march_madness.data.loaders import load_mens_data, load_womens_data
from march_madness.utils.data_processors import load_or_prepare_modeling_data
from march_madness.utils.helpers import train_and_predict_model
from march_madness.models.prediction import combine_predictions
from march_madness.features.matchup import create_tournament_prediction_dataset

def main():
    """Generate tournament predictions with optional real-world mode for 2025."""
    # Start timing
    start_time = time.time()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate March Madness predictions')
    parser.add_argument('--skip-data-load', action='store_true', 
                        help='Skip loading raw data (assume modeling data is cached)')
    parser.add_argument('--force-prepare', action='store_true',
                        help='Force data preparation even if cache exists')
    parser.add_argument('--tournament-only', action='store_true',
                        help='Only generate matchups between tournament teams (much faster)')
    parser.add_argument('--real-world', action='store_true',
                        help='Generate predictions for upcoming 2025 tournament (uses comprehensive cache)')
    parser.add_argument('--seasons', type=str, default=None,
                        help='Comma-separated list of seasons to predict (overrides defaults and real-world)')
    args = parser.parse_args()
    
    # Determine which seasons to predict
    if args.seasons:
        # User specified seasons override everything else
        prediction_seasons = [int(s) for s in args.seasons.split(',')]
        real_world_mode = False
    elif args.real_world:
        # Real-world mode includes 2025
        prediction_seasons = [2025]
        real_world_mode = True
        logger.info("REAL-WORLD MODE: Generating predictions for 2025 tournament")
    else:
        # Use default prediction seasons from config
        prediction_seasons = PREDICTION_SEASONS
        real_world_mode = False
    
    logger.info(f"Generating predictions for seasons: {prediction_seasons}")
    
    # Define directories
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, 'data')
    cache_dir = os.path.join(project_root, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    # Set cache file paths based on mode
    if real_world_mode:
        mens_cache_file = os.path.join(cache_dir, f"mens_all_data_{CURRENT_SEASON}_complete.pkl")
        womens_cache_file = os.path.join(cache_dir, f"womens_all_data_{CURRENT_SEASON}_complete.pkl")
        
        # Check if comprehensive cache files exist
        if not os.path.exists(mens_cache_file) or not os.path.exists(womens_cache_file):
            logger.warning(f"Comprehensive cache files not found. Please run preprocess_all_data.py first.")
            logger.warning(f"Falling back to standard cache files.")
            mens_cache_file = os.path.join(cache_dir, "men's_modeling_data.pkl")
            womens_cache_file = os.path.join(cache_dir, "women's_modeling_data.pkl")
    else:
        # Use standard cache files
        mens_cache_file = os.path.join(cache_dir, "men's_modeling_data.pkl")
        womens_cache_file = os.path.join(cache_dir, "women's_modeling_data.pkl") 
    
    # In real-world mode, we always use cached data
    if real_world_mode:
        args.skip_data_load = True
        logger.info("Real-world mode: Using cached data (--skip-data-load forced)")
    
    # Load or prepare modeling data
    mens_modeling_data = None
    womens_modeling_data = None
    
    if args.skip_data_load:
        logger.info("Skipping raw data loading, using cached modeling data...")
        # Load cached data directly
        if os.path.exists(mens_cache_file):
            logger.info(f"Loading men's cached data from {mens_cache_file}")
            with open(mens_cache_file, 'rb') as f:
                mens_modeling_data = pickle.load(f)
        else:
            logger.warning(f"Men's cache file not found: {mens_cache_file}")
            
        if os.path.exists(womens_cache_file):
            logger.info(f"Loading women's cached data from {womens_cache_file}")
            with open(womens_cache_file, 'rb') as f:
                womens_modeling_data = pickle.load(f)
        else:
            logger.warning(f"Women's cache file not found: {womens_cache_file}")
    else:
        # Load raw data
        logger.info("Loading raw data...")
        mens_data = load_mens_data(STARTING_SEASON, data_dir=data_dir)
        womens_data = load_womens_data(STARTING_SEASON, data_dir=data_dir)
        
        logger.info("Preparing modeling data...")
        # Get modeling data the standard way
        mens_modeling_data = load_or_prepare_modeling_data(
            mens_data, "men's", STARTING_SEASON, CURRENT_SEASON, 
            prediction_seasons,
            cache_dir=cache_dir,
            force_prepare=args.force_prepare
        )
        
        womens_modeling_data = load_or_prepare_modeling_data(
            womens_data, "women's", STARTING_SEASON, CURRENT_SEASON, 
            prediction_seasons,
            cache_dir=cache_dir,
            force_prepare=args.force_prepare
        )
    
    # Handle special setup for 2025 predictions
    if 2025 in prediction_seasons:
        # Check if we have 2025 seed data
        logger.info("Checking for 2025 tournament seeds...")
        mens_2025_seeds_path = os.path.join(data_dir, 'MNCAATourneySeeds2025.csv')
        womens_2025_seeds_path = os.path.join(data_dir, 'WNCAATourneySeeds2025.csv')
        
        # Process men's 2025 seeds if available
        if os.path.exists(mens_2025_seeds_path) and mens_modeling_data is not None:
            logger.info(f"Loading 2025 men's tournament seeds from {mens_2025_seeds_path}")
            mens_2025_seeds = pd.read_csv(mens_2025_seeds_path)
            
            # Add season column if not present
            if 'Season' not in mens_2025_seeds.columns:
                mens_2025_seeds['Season'] = 2025
            
            # Append 2025 seeds to existing seed data
            if 'df_seed' in mens_modeling_data:
                mens_modeling_data['df_seed'] = pd.concat([
                    mens_modeling_data['df_seed'], 
                    mens_2025_seeds
                ], ignore_index=True)
            else:
                logger.warning("No seed data found in men's modeling data")
                
            # Check if we need to generate matchups for 2025
            if args.tournament_only and 'season_matchups' in mens_modeling_data:
                if 2025 not in mens_modeling_data.get('season_matchups', {}):
                    logger.info("Generating tournament-only matchups for 2025 men's tournament")
                    
                    # Get required data with fallbacks
                    team_profiles = mens_modeling_data.get('enhanced_team_profiles', 
                                                    mens_modeling_data.get('team_profiles', pd.DataFrame()))
                    
                    # Check for tournament-only matchups first
                    if 'tournament_only_matchups' in mens_modeling_data and 2025 in mens_modeling_data['tournament_only_matchups']:
                        logger.info("Using pre-generated tournament-only matchups for 2025")
                        mens_modeling_data['season_matchups'][2025] = mens_modeling_data['tournament_only_matchups'][2025]
                    else:
                        # Create new matchups
                        try:
                            season_data = create_tournament_prediction_dataset(
                                [2025],
                                team_profiles,
                                mens_modeling_data['df_seed'],
                                mens_modeling_data.get('momentum_data', pd.DataFrame()),
                                mens_modeling_data.get('sos_data', pd.DataFrame()),
                                mens_modeling_data.get('coach_features', pd.DataFrame()),
                                mens_modeling_data.get('tourney_history', pd.DataFrame()),
                                mens_modeling_data.get('conf_strength', pd.DataFrame()),
                                mens_modeling_data.get('df_team_conferences', pd.DataFrame()),
                                mens_modeling_data.get('team_consistency', None),
                                mens_modeling_data.get('team_playstyle', None),
                                mens_modeling_data.get('round_performance', None),
                                mens_modeling_data.get('pressure_metrics', None),
                                mens_modeling_data.get('conf_impact', None),
                                mens_modeling_data.get('seed_features', None),
                                mens_modeling_data.get('coach_metrics', None),
                                tournament_only=True
                            )
                            mens_modeling_data['season_matchups'][2025] = season_data
                            logger.info(f"Generated {len(season_data)} tournament-only matchups for men's 2025")
                        except Exception as e:
                            logger.error(f"Error generating 2025 men's matchups: {str(e)}")
        else:
            logger.warning(f"2025 men's tournament seeds not found at {mens_2025_seeds_path}")
        
        # Process women's 2025 seeds if available
        if os.path.exists(womens_2025_seeds_path) and womens_modeling_data is not None:
            logger.info(f"Loading 2025 women's tournament seeds from {womens_2025_seeds_path}")
            womens_2025_seeds = pd.read_csv(womens_2025_seeds_path)
            
            # Add season column if not present
            if 'Season' not in womens_2025_seeds.columns:
                womens_2025_seeds['Season'] = 2025
            
            # Append 2025 seeds to existing seed data
            if 'df_seed' in womens_modeling_data:
                womens_modeling_data['df_seed'] = pd.concat([
                    womens_modeling_data['df_seed'], 
                    womens_2025_seeds
                ], ignore_index=True)
            else:
                logger.warning("No seed data found in women's modeling data")
                
            # Check if we need to generate matchups for 2025
            if args.tournament_only and 'season_matchups' in womens_modeling_data:
                if 2025 not in womens_modeling_data.get('season_matchups', {}):
                    logger.info("Generating tournament-only matchups for 2025 women's tournament")
                    
                    # Get required data with fallbacks
                    team_profiles = womens_modeling_data.get('enhanced_team_profiles', 
                                                      womens_modeling_data.get('team_profiles', pd.DataFrame()))
                    
                    # Check for tournament-only matchups first
                    if 'tournament_only_matchups' in womens_modeling_data and 2025 in womens_modeling_data['tournament_only_matchups']:
                        logger.info("Using pre-generated tournament-only matchups for 2025")
                        womens_modeling_data['season_matchups'][2025] = womens_modeling_data['tournament_only_matchups'][2025]
                    else:
                        # Create new matchups
                        try:
                            season_data = create_tournament_prediction_dataset(
                                [2025],
                                team_profiles,
                                womens_modeling_data['df_seed'],
                                womens_modeling_data.get('momentum_data', pd.DataFrame()),
                                womens_modeling_data.get('sos_data', pd.DataFrame()),
                                womens_modeling_data.get('coach_features', pd.DataFrame()),
                                womens_modeling_data.get('tourney_history', pd.DataFrame()),
                                womens_modeling_data.get('conf_strength', pd.DataFrame()),
                                womens_modeling_data.get('df_team_conferences', pd.DataFrame()),
                                womens_modeling_data.get('team_consistency', None),
                                womens_modeling_data.get('team_playstyle', None),
                                womens_modeling_data.get('round_performance', None),
                                womens_modeling_data.get('pressure_metrics', None),
                                womens_modeling_data.get('conf_impact', None),
                                womens_modeling_data.get('seed_features', None),
                                womens_modeling_data.get('coach_metrics', None),
                                tournament_only=True
                            )
                            womens_modeling_data['season_matchups'][2025] = season_data
                            logger.info(f"Generated {len(season_data)} tournament-only matchups for women's 2025")
                        except Exception as e:
                            logger.error(f"Error generating 2025 women's matchups: {str(e)}")
        else:
            logger.warning(f"2025 women's tournament seeds not found at {womens_2025_seeds_path}")
    
    # Apply tournament-only filtering for non-2025 seasons if needed
    if args.tournament_only and not real_world_mode:
        # Similar to tournament_teams_predictions.py logic
        # This would regenerate tournament-only matchups for historical seasons
        pass
    
    # Load trained models and metadata
    logger.info("Loading trained models...")
    models_dir = os.path.join(project_root, 'models')
    
    try:
        with open(os.path.join(models_dir, 'mens_model.pkl'), 'rb') as f:
            mens_model = pickle.load(f)
        
        with open(os.path.join(models_dir, 'mens_metadata.pkl'), 'rb') as f:
            mens_metadata = pickle.load(f)
        
        with open(os.path.join(models_dir, 'womens_model.pkl'), 'rb') as f:
            womens_model = pickle.load(f)
        
        with open(os.path.join(models_dir, 'womens_metadata.pkl'), 'rb') as f:
            womens_metadata = pickle.load(f)
    except FileNotFoundError as e:
        logger.error(f"Could not load model files: {str(e)}")
        return
    
    # Generate predictions
    mens_predictions = pd.DataFrame()
    womens_predictions = pd.DataFrame()
    
    if mens_modeling_data is not None:
        logger.info("Generating Men's predictions...")
        mens_predictions = train_and_predict_model(
            mens_modeling_data, "men's", [], None, prediction_seasons,
            model=mens_model,
            feature_cols=mens_metadata['feature_cols'],
            scaler=mens_metadata['scaler'],
            dropped_features=mens_metadata['dropped_features']
        )
        logger.info(f"Generated {len(mens_predictions)} men's predictions")
    else:
        logger.warning("Skipping men's predictions due to missing data")
    
    if womens_modeling_data is not None:
        logger.info("Generating Women's predictions...")
        womens_predictions = train_and_predict_model(
            womens_modeling_data, "women's", [], None, prediction_seasons,
            model=womens_model,
            feature_cols=womens_metadata['feature_cols'],
            scaler=womens_metadata['scaler'],
            dropped_features=womens_metadata['dropped_features']
        )
        logger.info(f"Generated {len(womens_predictions)} women's predictions")
    else:
        logger.warning("Skipping women's predictions due to missing data")
    
    # Combine predictions
    if not mens_predictions.empty or not womens_predictions.empty:
        logger.info("Combining predictions...")
        combined_submission = combine_predictions(mens_predictions, womens_predictions)
        
        # Create submission filename
        parts = []
        if args.tournament_only:
            parts.append("tournament_only")
        if real_world_mode:
            parts.append("2025")
        if parts:
            suffix = "_" + "_".join(parts)
        else:
            suffix = ""
            
        submission_path = os.path.join(project_root, f"submission{suffix}.csv")
        combined_submission.to_csv(submission_path, index=False)
        
        # Report results
        logger.info(f"Predictions generated and saved to '{submission_path}'")
        logger.info(f"Total predictions: {len(combined_submission)}")
        logger.info(f"Men's predictions: {len(mens_predictions)}")
        logger.info(f"Women's predictions: {len(womens_predictions)}")
    else:
        logger.error("No predictions were generated!")
    
    # Report timing
    end_time = time.time()
    elapsed = end_time - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    logger.info(f"Total execution time: {minutes} minutes, {seconds} seconds")

if __name__ == "__main__":
    main()