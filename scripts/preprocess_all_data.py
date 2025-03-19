#!/usr/bin/env python
"""
Comprehensive Data Preprocessing Script for March Madness Predictor

This script processes ALL historical NCAA basketball data and caches it for future use.
Run this once to create a complete cached dataset that can be used for 2025 predictions
without needing to reprocess historical data.
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
import time
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the package to path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from march_madness.config import STARTING_SEASON, CURRENT_SEASON
from march_madness.data.loaders import load_mens_data, load_womens_data
from march_madness.utils.data_processors import prepare_modeling_data
from march_madness.features.matchup import create_tournament_prediction_dataset

def main():
    """Process and cache all historical NCAA tournament data."""
    start_time = time.time()
    
    # Create timestamp for logging
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Starting comprehensive data preprocessing at {timestamp}")
    
    # Define all seasons to process (all historical seasons + current season)
    # We'll exclude 2020 since that tournament was canceled due to COVID-19
    all_seasons = [year for year in range(STARTING_SEASON, CURRENT_SEASON + 1) if year != 2020]
    logger.info(f"Processing data for seasons: {all_seasons}")
    
    # Create cache directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cache_dir = os.path.join(project_root, 'cache')
    data_dir = os.path.join(project_root, 'data')
    os.makedirs(cache_dir, exist_ok=True)
    
    # Define cache files with timestamp to avoid conflicts
    mens_cache_file = os.path.join(cache_dir, f"mens_all_data_{CURRENT_SEASON}_complete.pkl")
    womens_cache_file = os.path.join(cache_dir, f"womens_all_data_{CURRENT_SEASON}_complete.pkl")
    
    # Process men's data
    logger.info("Loading men's raw data...")
    mens_data = load_mens_data(STARTING_SEASON, data_dir=data_dir)
    
    logger.info("Processing men's tournament data...")
    # Set current_season as validation_season to avoid None errors
    # We'll process all seasons at once
    mens_modeling_data = prepare_modeling_data(
        mens_data, "men's", STARTING_SEASON, CURRENT_SEASON, 
        all_seasons,
        cache_dir=None  # Don't use existing cache since we're creating fresh cache
    )
    
    # Save men's data
    logger.info(f"Saving men's data to {mens_cache_file}")
    with open(mens_cache_file, 'wb') as f:
        pickle.dump(mens_modeling_data, f)
    
    # Process women's data
    logger.info("Loading women's raw data...")
    womens_data = load_womens_data(STARTING_SEASON, data_dir=data_dir)
    
    logger.info("Processing women's tournament data...")
    womens_modeling_data = prepare_modeling_data(
        womens_data, "women's", STARTING_SEASON, CURRENT_SEASON, 
        all_seasons,
        cache_dir=None  # Don't use existing cache
    )
    
    # Save women's data
    logger.info(f"Saving women's data to {womens_cache_file}")
    with open(womens_cache_file, 'wb') as f:
        pickle.dump(womens_modeling_data, f)
    
    # Add tournament-only matchups (for faster prediction generation later)
    logger.info("Generating tournament-only matchups for faster future predictions...")
    
    # Iterate through all prediction seasons
    prediction_seasons = [year for year in range(2021, CURRENT_SEASON + 1)]
    
    # Process men's tournament-only matchups
    for season in prediction_seasons:
        try:
            logger.info(f"Generating men's tournament-only matchups for season {season}")
            
            # Skip if season_matchups doesn't exist for this season
            if 'season_matchups' not in mens_modeling_data or season not in mens_modeling_data['season_matchups']:
                logger.warning(f"No matchup data found for men's season {season}, skipping")
                continue
                
            # Get all required data for tournament-only matchup generation
            team_profiles = mens_modeling_data.get('enhanced_team_profiles', pd.DataFrame())
            if team_profiles.empty:
                # Try alternative key name if enhanced_team_profiles doesn't exist
                team_profiles = mens_modeling_data.get('team_profiles', pd.DataFrame())
                
            # If we still don't have team profiles, skip this season
            if team_profiles.empty:
                logger.warning(f"No team profiles found for men's season {season}, skipping")
                continue
                
            # Get seed data
            seed_data = mens_modeling_data.get('df_seed', pd.DataFrame())
            season_seed_data = seed_data[seed_data['Season'] == season]
            
            # Create tournament-only matchups
            tournament_teams = season_seed_data['TeamID'].unique()
            
            if len(tournament_teams) == 0:
                logger.warning(f"No tournament teams found for men's season {season}, skipping")
                continue
                
            logger.info(f"Found {len(tournament_teams)} men's tournament teams for season {season}")
            
            # Get required fields with graceful fallbacks to empty DataFrames
            season_data = create_tournament_prediction_dataset(
                [season],
                team_profiles,
                seed_data,
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
                tournament_only=True  # Only generate matchups between tournament teams
            )
            
            # Store tournament-only matchups in a new dictionary
            if 'tournament_only_matchups' not in mens_modeling_data:
                mens_modeling_data['tournament_only_matchups'] = {}
                
            mens_modeling_data['tournament_only_matchups'][season] = season_data
            logger.info(f"Generated {len(season_data)} men's tournament-only matchups for season {season}")
            
        except Exception as e:
            logger.error(f"Error generating men's tournament-only matchups for season {season}: {str(e)}")
    
    # Process women's tournament-only matchups
    for season in prediction_seasons:
        try:
            logger.info(f"Generating women's tournament-only matchups for season {season}")
            
            # Skip if season_matchups doesn't exist for this season
            if 'season_matchups' not in womens_modeling_data or season not in womens_modeling_data['season_matchups']:
                logger.warning(f"No matchup data found for women's season {season}, skipping")
                continue
                
            # Get all required data for tournament-only matchup generation
            team_profiles = womens_modeling_data.get('enhanced_team_profiles', pd.DataFrame())
            if team_profiles.empty:
                # Try alternative key name if enhanced_team_profiles doesn't exist
                team_profiles = womens_modeling_data.get('team_profiles', pd.DataFrame())
                
            # If we still don't have team profiles, skip this season
            if team_profiles.empty:
                logger.warning(f"No team profiles found for women's season {season}, skipping")
                continue
                
            # Get seed data
            seed_data = womens_modeling_data.get('df_seed', pd.DataFrame())
            season_seed_data = seed_data[seed_data['Season'] == season]
            
            # Create tournament-only matchups
            tournament_teams = season_seed_data['TeamID'].unique()
            
            if len(tournament_teams) == 0:
                logger.warning(f"No tournament teams found for women's season {season}, skipping")
                continue
                
            logger.info(f"Found {len(tournament_teams)} women's tournament teams for season {season}")
            
            # Get required fields with graceful fallbacks to empty DataFrames
            season_data = create_tournament_prediction_dataset(
                [season],
                team_profiles,
                seed_data,
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
                tournament_only=True  # Only generate matchups between tournament teams
            )
            
            # Store tournament-only matchups in a new dictionary
            if 'tournament_only_matchups' not in womens_modeling_data:
                womens_modeling_data['tournament_only_matchups'] = {}
                
            womens_modeling_data['tournament_only_matchups'][season] = season_data
            logger.info(f"Generated {len(season_data)} women's tournament-only matchups for season {season}")
            
        except Exception as e:
            logger.error(f"Error generating women's tournament-only matchups for season {season}: {str(e)}")
    
    # Save final combined data with tournament-only matchups
    logger.info(f"Saving final men's data with tournament-only matchups to {mens_cache_file}")
    with open(mens_cache_file, 'wb') as f:
        pickle.dump(mens_modeling_data, f)
        
    logger.info(f"Saving final women's data with tournament-only matchups to {womens_cache_file}")
    with open(womens_cache_file, 'wb') as f:
        pickle.dump(womens_modeling_data, f)
    
    # Create symlinks to the standard cache filenames for easy loading
    mens_std_cache = os.path.join(cache_dir, "men's_modeling_data.pkl")
    womens_std_cache = os.path.join(cache_dir, "women's_modeling_data.pkl")
    
    # Remove existing symlinks/files if they exist
    if os.path.exists(mens_std_cache):
        os.remove(mens_std_cache)
    if os.path.exists(womens_std_cache):
        os.remove(womens_std_cache)
    
    # Create copies instead of symlinks for better compatibility
    logger.info(f"Creating standard cache files for easy loading")
    with open(mens_std_cache, 'wb') as f:
        pickle.dump(mens_modeling_data, f)
        
    with open(womens_std_cache, 'wb') as f:
        pickle.dump(womens_modeling_data, f)
    
    # Report completion and timing
    end_time = time.time()
    elapsed = end_time - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    
    logger.info(f"All data processing completed successfully!")
    logger.info(f"Total processing time: {minutes} minutes, {seconds} seconds")
    logger.info(f"Cached data ready for 2025 predictions")

if __name__ == "__main__":
    main()