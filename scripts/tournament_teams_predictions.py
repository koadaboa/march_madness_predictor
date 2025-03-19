#!/usr/bin/env python
"""
Generate NCAA Tournament predictions with the option to only create matchups
between tournament teams, significantly improving performance.
"""

import os
import pandas as pd
import numpy as np
import pickle
import sys
import time
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
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
    # Start timing
    start_time = time.time()
    logger.info("Tournament teams prediction script started")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate March Madness predictions with tournament-only option')
    parser.add_argument('--tournament-only', action='store_true', 
                        help='Only generate matchups between tournament teams (much faster)')
    parser.add_argument('--skip-data-load', action='store_true', 
                        help='Skip loading raw data (assume modeling data is cached)')
    parser.add_argument('--force-prepare', action='store_true',
                        help='Force data preparation even if cache exists')
    parser.add_argument('--seasons', type=str, default=','.join(map(str, PREDICTION_SEASONS)),
                        help='Comma-separated list of seasons to predict')
    args = parser.parse_args()
    
    # Parse seasons
    prediction_seasons = [int(s) for s in args.seasons.split(',')]
    
    logger.info("Generating NCAA Basketball Tournament Predictions")
    if args.tournament_only:
        logger.info("TOURNAMENT-ONLY MODE: Only generating matchups between tournament teams")
    
    # Define directories
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, 'data')
    cache_dir = os.path.join(project_root, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    # Load data if needed
    if args.skip_data_load:
        logger.info("Skipping raw data loading, using cached modeling data...")
        mens_data = womens_data = None
    else:
        # Load raw data
        logger.info("Loading raw data...")
        mens_data = load_mens_data(STARTING_SEASON, data_dir=data_dir)
        womens_data = load_womens_data(STARTING_SEASON, data_dir=data_dir)
    
    # Get original modeling data from cache or prepare it
    logger.info("Getting modeling data for men's tournament...")
    mens_modeling_data = load_or_prepare_modeling_data(
        mens_data, "men's", STARTING_SEASON, CURRENT_SEASON, 
        prediction_seasons,
        cache_dir=cache_dir,
        force_prepare=args.force_prepare
    )
    
    # Check if we have the season_matchups key, which is critical
    if 'season_matchups' not in mens_modeling_data:
        logger.error("Cached modeling data missing 'season_matchups' key. Cannot proceed.")
        return
    
    # Check if seed data is available
    if 'df_seed' not in mens_modeling_data:
        logger.error("Cached modeling data missing 'df_seed' key. Cannot proceed.")
        return
        
    # List all keys in modeling data for debugging
    logger.info(f"Available keys in mens_modeling_data: {list(mens_modeling_data.keys())}")
    
    # Modify the season_matchups data using our tournament_only parameter
    if args.tournament_only:
        logger.info("Regenerating men's matchups with tournament_only=True...")
        for season in prediction_seasons:
            if season in mens_modeling_data.get('season_matchups', {}):
                logger.info(f"Regenerating matchups for men's season {season}...")
                
                # Get required data, with fallbacks to empty DataFrames when keys don't exist
                team_profiles = mens_modeling_data.get('enhanced_team_profiles', 
                                                     mens_modeling_data.get('team_profiles', pd.DataFrame()))
                seed_data = mens_modeling_data['df_seed']
                momentum_data = mens_modeling_data.get('momentum_data', pd.DataFrame())
                sos_data = mens_modeling_data.get('sos_data', pd.DataFrame())
                coach_features = mens_modeling_data.get('coach_features', pd.DataFrame())
                tourney_history = mens_modeling_data.get('tourney_history', pd.DataFrame())
                conf_strength = mens_modeling_data.get('conf_strength', pd.DataFrame())
                team_conferences = mens_modeling_data.get('df_team_conferences', pd.DataFrame())
                team_consistency = mens_modeling_data.get('team_consistency', None)
                team_playstyle = mens_modeling_data.get('team_playstyle', None)
                round_performance = mens_modeling_data.get('round_performance', None)
                pressure_metrics = mens_modeling_data.get('pressure_metrics', None)
                conf_impact = mens_modeling_data.get('conf_impact', None)
                seed_features = mens_modeling_data.get('seed_features', None)
                coach_metrics = mens_modeling_data.get('coach_metrics', None)
                
                # Check if team profiles exist
                if team_profiles.empty:
                    logger.error(f"No team profiles found for season {season}. Skipping matchup generation.")
                    continue
                
                # Create new matchups using tournament_only=True
                try:
                    season_data = create_tournament_prediction_dataset(
                        [season],
                        team_profiles,
                        seed_data,
                        momentum_data,
                        sos_data,
                        coach_features,
                        tourney_history,
                        conf_strength,
                        team_conferences,
                        team_consistency,
                        team_playstyle,
                        round_performance,
                        pressure_metrics,
                        conf_impact,
                        seed_features,
                        coach_metrics,
                        tournament_only=True  # This is the key parameter!
                    )
                    
                    # Replace the existing season matchups with our tournament-only version
                    mens_modeling_data['season_matchups'][season] = season_data
                    logger.info(f"  Reduced to {len(season_data)} tournament-only matchups")
                    
                except Exception as e:
                    logger.error(f"Error generating tournament matchups: {str(e)}", exc_info=True)
                    continue
    
    # Same process for women's tournament
    logger.info("Getting modeling data for women's tournament...")
    womens_modeling_data = load_or_prepare_modeling_data(
        womens_data, "women's", STARTING_SEASON, CURRENT_SEASON, 
        prediction_seasons,
        cache_dir=cache_dir,
        force_prepare=args.force_prepare
    )
    
    # Check if we have the season_matchups key for women
    if 'season_matchups' not in womens_modeling_data:
        logger.error("Women's cached modeling data missing 'season_matchups' key. Skipping women's predictions.")
    elif args.tournament_only:
        logger.info("Regenerating women's matchups with tournament_only=True...")
        for season in prediction_seasons:
            if season in womens_modeling_data.get('season_matchups', {}):
                logger.info(f"Regenerating matchups for women's season {season}...")
                
                # Get required data, with fallbacks
                team_profiles = womens_modeling_data.get('enhanced_team_profiles', 
                                                      womens_modeling_data.get('team_profiles', pd.DataFrame()))
                seed_data = womens_modeling_data.get('df_seed', pd.DataFrame())
                momentum_data = womens_modeling_data.get('momentum_data', pd.DataFrame())
                sos_data = womens_modeling_data.get('sos_data', pd.DataFrame())
                coach_features = womens_modeling_data.get('coach_features', pd.DataFrame())
                tourney_history = womens_modeling_data.get('tourney_history', pd.DataFrame())
                conf_strength = womens_modeling_data.get('conf_strength', pd.DataFrame())
                team_conferences = womens_modeling_data.get('df_team_conferences', pd.DataFrame())
                team_consistency = womens_modeling_data.get('team_consistency', None)
                team_playstyle = womens_modeling_data.get('team_playstyle', None)
                round_performance = womens_modeling_data.get('round_performance', None)
                pressure_metrics = womens_modeling_data.get('pressure_metrics', None)
                conf_impact = womens_modeling_data.get('conf_impact', None)
                seed_features = womens_modeling_data.get('seed_features', None)
                coach_metrics = womens_modeling_data.get('coach_metrics', None)
                
                # Check if team profiles exist
                if team_profiles.empty:
                    logger.error(f"No women's team profiles found for season {season}. Skipping matchup generation.")
                    continue
                
                # Create new matchups using tournament_only=True
                try:
                    season_data = create_tournament_prediction_dataset(
                        [season],
                        team_profiles,
                        seed_data,
                        momentum_data,
                        sos_data,
                        coach_features,
                        tourney_history,
                        conf_strength,
                        team_conferences,
                        team_consistency,
                        team_playstyle,
                        round_performance,
                        pressure_metrics,
                        conf_impact,
                        seed_features,
                        coach_metrics,
                        tournament_only=True  # This is the key parameter!
                    )
                    
                    # Replace the existing season matchups with our tournament-only version
                    womens_modeling_data['season_matchups'][season] = season_data
                    logger.info(f"  Reduced to {len(season_data)} tournament-only matchups")
                    
                except Exception as e:
                    logger.error(f"Error generating women's tournament matchups: {str(e)}", exc_info=True)
                    continue
    
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
    logger.info("Generating Men's predictions...")
    mens_predictions = train_and_predict_model(
        mens_modeling_data, "men's", [], None, prediction_seasons,
        model=mens_model,
        feature_cols=mens_metadata['feature_cols'],
        scaler=mens_metadata['scaler'],
        dropped_features=mens_metadata['dropped_features']
    )
    
    # Check if womens_modeling_data has season_matchups
    if 'season_matchups' in womens_modeling_data:
        logger.info("Generating Women's predictions...")
        womens_predictions = train_and_predict_model(
            womens_modeling_data, "women's", [], None, prediction_seasons,
            model=womens_model,
            feature_cols=womens_metadata['feature_cols'],
            scaler=womens_metadata['scaler'],
            dropped_features=womens_metadata['dropped_features']
        )
    else:
        logger.warning("Skipping women's predictions due to missing season_matchups")
        womens_predictions = pd.DataFrame()
    
    # Combine predictions if both exist
    if not mens_predictions.empty and not womens_predictions.empty:
        logger.info("Combining predictions...")
        combined_submission = combine_predictions(mens_predictions, womens_predictions)
    elif not mens_predictions.empty:
        logger.info("Using only men's predictions...")
        combined_submission = mens_predictions.rename(columns={'Team1ID': 'ID', 'Pred': 'Pred'})
    elif not womens_predictions.empty:
        logger.info("Using only women's predictions...")
        combined_submission = womens_predictions.rename(columns={'Team1ID': 'ID', 'Pred': 'Pred'})
    else:
        logger.error("No predictions generated!")
        return
    
    # Save predictions
    suffix = "_tournament_only" if args.tournament_only else ""
    submission_path = os.path.join(project_root, f"submission{suffix}.csv")
    combined_submission.to_csv(submission_path, index=False)
    
    # Report results
    logger.info(f"Predictions generated and saved to '{submission_path}'")
    logger.info(f"Total predictions: {len(combined_submission)}")
    logger.info(f"Men's predictions: {len(mens_predictions)}")
    logger.info(f"Women's predictions: {len(womens_predictions)}")
    
    # Report timing
    end_time = time.time()
    elapsed = end_time - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    logger.info(f"Total execution time: {minutes} minutes, {seconds} seconds")


if __name__ == "__main__":
    main()