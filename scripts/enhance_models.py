#!/usr/bin/env python
"""
Script to enhance NCAA Basketball Tournament prediction models with 
round-specific training and improved upset detection

This script enhances your existing models using the new specialized modeling components.
It preserves your pipeline structure while significantly improving prediction accuracy,
especially for late-round games and upset detection.

Usage:
    python enhance_models.py

The script will:
1. Load your existing models and data
2. Train enhanced versions with round-specific components
3. Save the enhanced models to the models directory
4. Generate predictions with the enhanced models
5. Evaluate and compare performance 
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
import argparse
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the package to path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from march_madness.config import (STARTING_SEASON, CURRENT_SEASON, 
                                 TRAINING_SEASONS, VALIDATION_SEASON, 
                                 PREDICTION_SEASONS)
from march_madness.utils.data_processors import load_or_prepare_modeling_data
from march_madness.models.prediction import combine_predictions
from march_madness.models.integration import (enhance_existing_models,
                                            predict_with_enhanced_model,
                                            load_enhanced_model)
from march_madness.models.evaluation import evaluate_predictions_against_actual

def main():
    """
    Main function to enhance models and generate improved predictions
    """
    start_time = datetime.now()
    logger.info(f"Starting model enhancement at {start_time}")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Enhance NCAA Tournament prediction models')
    parser.add_argument('--skip-training', action='store_true', 
                        help='Skip enhancing models (use existing enhanced models if available)')
    parser.add_argument('--prediction-seasons', type=str, default=None,
                        help='Comma-separated list of seasons to predict (default: 2023,2024)')
    parser.add_argument('--compare', action='store_true',
                        help='Generate and compare predictions from both original and enhanced models')
    parser.add_argument('--debug', action='store_true', 
                        help='Enable verbose debug output')
    args = parser.parse_args()
    
    # Set debug level if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
    
    # Setup directories
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, 'data')
    cache_dir = os.path.join(project_root, 'cache')
    models_dir = os.path.join(project_root, 'models')
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # Determine which seasons to predict
    if args.prediction_seasons:
        prediction_seasons = [int(s) for s in args.prediction_seasons.split(',')]
    else:
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
        logger.error("Please run data_processing.py first to generate required data")
        return

    # Women's modeling data
    womens_cache_file = os.path.join(cache_dir, "women's_modeling_data.pkl")
    if os.path.exists(womens_cache_file):
        with open(womens_cache_file, 'rb') as f:
            womens_modeling_data = pickle.load(f)
        logger.info(f"Loaded women's modeling data from {womens_cache_file}")
    else:
        logger.error(f"Women's modeling data not found at {womens_cache_file}")
        logger.error("Please run data_processing.py first to generate required data")
        return
    
    # Examine the modeling data
    if args.debug:
        # Look at seasons in tournament data
        mens_tourney = mens_modeling_data.get('tourney_data', pd.DataFrame())
        if not mens_tourney.empty:
            logger.debug(f"Men's tournament data seasons: {sorted(mens_tourney['Season'].unique())}")
            logger.debug(f"Men's tournament data shape: {mens_tourney.shape}")
        
        # Look at season_matchups
        mens_seasons = mens_modeling_data.get('season_matchups', {}).keys()
        logger.debug(f"Men's season_matchups keys: {sorted(mens_seasons)}")
        
        # Look at first season's matchups
        if mens_seasons:
            first_season = sorted(mens_seasons)[0]
            first_matchups = mens_modeling_data['season_matchups'][first_season]
            logger.debug(f"Men's first season {first_season} matchups shape: {first_matchups.shape}")
            logger.debug(f"Men's first season {first_season} matchups columns: {first_matchups.columns.tolist()}")
    
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
        
        if args.debug:
            logger.debug(f"Men's model type: {type(mens_model)}")
            logger.debug(f"Men's metadata keys: {mens_metadata.keys()}")
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
    
    # Check for existing enhanced models or create them
    mens_enhanced_file = os.path.join(models_dir, 'mens_enhanced_model.pkl')
    womens_enhanced_file = os.path.join(models_dir, 'womens_enhanced_model.pkl')
    
    enhanced_models = {}
    
    if args.skip_training and os.path.exists(mens_enhanced_file) and os.path.exists(womens_enhanced_file):
        logger.info("Loading existing enhanced models...")
        
        # Load enhanced men's model
        mens_enhanced, mens_enhanced_features = load_enhanced_model(mens_enhanced_file)
        enhanced_models['mens_model'] = mens_enhanced
        enhanced_models['mens_feature_cols'] = mens_enhanced_features
        logger.info("Loaded enhanced men's model")
        
        # Load enhanced women's model
        womens_enhanced, womens_enhanced_features = load_enhanced_model(womens_enhanced_file)
        enhanced_models['womens_model'] = womens_enhanced
        enhanced_models['womens_feature_cols'] = womens_enhanced_features
        logger.info("Loaded enhanced women's model")
    else:
        logger.info("Creating enhanced models...")
        
        # Enhance existing models
        enhanced_models = enhance_existing_models(
            mens_model, mens_metadata,
            womens_model, womens_metadata,
            mens_modeling_data, womens_modeling_data,
            output_dir=models_dir
        )
        
        if enhanced_models:
            logger.info("Enhanced models created and saved")
        else:
            logger.warning("No enhanced models were created")
    
    # Generate predictions with enhanced models
    if 'mens_model' in enhanced_models or 'womens_model' in enhanced_models:
        logger.info("Generating predictions with enhanced models...")
        
        # Initialize prediction DataFrames
        mens_enhanced_predictions = []
        womens_enhanced_predictions = []
        
        # Process each prediction season
        for season in prediction_seasons:
            # Men's predictions if model exists
            if 'mens_model' in enhanced_models and season in mens_modeling_data['season_matchups']:
                logger.info(f"Generating enhanced men's predictions for season {season}")
                
                # Get matchups for this season
                season_matchups = mens_modeling_data['season_matchups'][season]
                
                # Prepare features
                X_predict = season_matchups.drop(['Season', 'Team1ID', 'Team2ID', 'MatchupID'], 
                                              axis=1, errors='ignore')
                # Also drop Pred if it exists
                if 'Pred' in X_predict.columns:
                    X_predict = X_predict.drop('Pred', axis=1)
                
                # Generate predictions
                enhanced_preds = predict_with_enhanced_model(
                    enhanced_models['mens_model'],
                    X_predict,
                    gender="men's"
                )
                
                # Create prediction DataFrame
                season_preds = season_matchups[['Season', 'Team1ID', 'Team2ID', 'MatchupID']].copy()
                season_preds['Pred'] = enhanced_preds
                
                mens_enhanced_predictions.append(season_preds)
                logger.info(f"Generated {len(season_preds)} enhanced men's predictions for season {season}")
            else:
                if 'mens_model' not in enhanced_models:
                    logger.warning(f"No enhanced men's model was created")
                else:
                    logger.warning(f"No men's matchup data found for season {season}")
            
            # Women's predictions if model exists
            if 'womens_model' in enhanced_models and season in womens_modeling_data['season_matchups']:
                logger.info(f"Generating enhanced women's predictions for season {season}")
                
                # Get matchups for this season
                season_matchups = womens_modeling_data['season_matchups'][season]
                
                # Prepare features
                X_predict = season_matchups.drop(['Season', 'Team1ID', 'Team2ID', 'MatchupID'], 
                                              axis=1, errors='ignore')
                # Also drop Pred if it exists
                if 'Pred' in X_predict.columns:
                    X_predict = X_predict.drop('Pred', axis=1)
                
                # Generate predictions
                enhanced_preds = predict_with_enhanced_model(
                    enhanced_models['womens_model'],
                    X_predict,
                    gender="women's"
                )
                
                # Create prediction DataFrame
                season_preds = season_matchups[['Season', 'Team1ID', 'Team2ID', 'MatchupID']].copy()
                season_preds['Pred'] = enhanced_preds
                
                womens_enhanced_predictions.append(season_preds)
                logger.info(f"Generated {len(season_preds)} enhanced women's predictions for season {season}")
            else:
                if 'womens_model' not in enhanced_models:
                    logger.warning(f"No enhanced women's model was created")
                else:
                    logger.warning(f"No women's matchup data found for season {season}")
        
        # Combine all prediction seasons
        if mens_enhanced_predictions:
            mens_enhanced_df = pd.concat(mens_enhanced_predictions, ignore_index=True)
            logger.info(f"Total enhanced men's predictions: {len(mens_enhanced_df)}")
        else:
            mens_enhanced_df = pd.DataFrame(columns=['Season', 'Team1ID', 'Team2ID', 'MatchupID', 'Pred'])
            logger.warning("No enhanced men's predictions generated")
        
        if womens_enhanced_predictions:
            womens_enhanced_df = pd.concat(womens_enhanced_predictions, ignore_index=True)
            logger.info(f"Total enhanced women's predictions: {len(womens_enhanced_df)}")
        else:
            womens_enhanced_df = pd.DataFrame(columns=['Season', 'Team1ID', 'Team2ID', 'MatchupID', 'Pred'])
            logger.warning("No enhanced women's predictions generated")
        
        # Create submission file with enhanced predictions
        enhanced_submission = combine_predictions(mens_enhanced_df, womens_enhanced_df)
        enhanced_submission_path = os.path.join(project_root, "submission_enhanced.csv")
        enhanced_submission.to_csv(enhanced_submission_path, index=False)
        logger.info(f"Enhanced submission file saved to {enhanced_submission_path}")
        
        # Generate original predictions for comparison if requested
        if args.compare:
            from march_madness.utils.helpers import train_and_predict_model
            
            logger.info("Generating predictions with original models for comparison...")
            
            # Men's original predictions
            mens_original_preds = train_and_predict_model(
                mens_modeling_data, "men's", [], None, prediction_seasons,
                model=mens_model,
                feature_cols=mens_metadata['feature_cols'],
                scaler=mens_metadata['scaler'],
                dropped_features=mens_metadata['dropped_features']
            )
            
            # Women's original predictions
            womens_original_preds = train_and_predict_model(
                womens_modeling_data, "women's", [], None, prediction_seasons,
                model=womens_model,
                feature_cols=womens_metadata['feature_cols'],
                scaler=womens_metadata['scaler'],
                dropped_features=womens_metadata['dropped_features']
            )
            
            # Create submission file with original predictions
            original_submission = combine_predictions(mens_original_preds, womens_original_preds)
            original_submission_path = os.path.join(project_root, "submission_original.csv")
            original_submission.to_csv(original_submission_path, index=False)
            logger.info(f"Original submission file saved to {original_submission_path}")
            
            # Compare with actual tournament results
            if prediction_seasons:
                logger.info("Evaluating and comparing predictions against actual tournament results...")
                
                # Load men's tournament results
                mens_results_path = os.path.join(data_dir, 'MNCAATourneyDetailedResults.csv')
                if os.path.exists(mens_results_path):
                    mens_results = pd.read_csv(mens_results_path)
                    mens_results = mens_results[mens_results['Season'].isin(prediction_seasons)]
                    
                    if not mens_results.empty:
                        logger.info(f"Found {len(mens_results)} men's tournament games for seasons {prediction_seasons}")
                        
                        # Evaluate original model
                        logger.info("Evaluating original men's model predictions...")
                        original_evaluation = evaluate_predictions_against_actual(
                            mens_original_preds, 
                            mens_results,
                            gender="men's"
                        )
                        
                        # Evaluate enhanced model
                        logger.info("Evaluating enhanced men's model predictions...")
                        enhanced_evaluation = evaluate_predictions_against_actual(
                            mens_enhanced_df,
                            mens_results,
                            gender="men's"
                        )
                        
                        # Compare results
                        logger.info("\n===== COMPARISON: ORIGINAL vs ENHANCED MEN'S MODEL =====")
                        logger.info(f"Original Model - Overall Accuracy: {original_evaluation['accuracy']:.4f}")
                        logger.info(f"Enhanced Model - Overall Accuracy: {enhanced_evaluation['accuracy']:.4f}")
                        
                        accuracy_diff = enhanced_evaluation['accuracy'] - original_evaluation['accuracy']
                        logger.info(f"Accuracy Improvement: {accuracy_diff:.4f} ({accuracy_diff * 100:.2f}%)")
                        
                        logger.info(f"Original Model - Brier Score: {original_evaluation['brier_score']:.4f}")
                        logger.info(f"Enhanced Model - Brier Score: {enhanced_evaluation['brier_score']:.4f}")
                        
                        brier_diff = original_evaluation['brier_score'] - enhanced_evaluation['brier_score']
                        logger.info(f"Brier Score Improvement: {brier_diff:.4f} ({brier_diff / original_evaluation['brier_score'] * 100:.2f}%)")
                        
                        # Compare upset detection
                        if 'results_df' in original_evaluation and 'results_df' in enhanced_evaluation:
                            # Calculate upset accuracy
                            orig_upsets = original_evaluation['results_df'][original_evaluation['results_df']['Upset']]
                            enhanced_upsets = enhanced_evaluation['results_df'][enhanced_evaluation['results_df']['Upset']]
                            
                            if len(orig_upsets) > 0 and len(enhanced_upsets) > 0:
                                orig_upset_acc = orig_upsets['Correct'].mean()
                                enhanced_upset_acc = enhanced_upsets['Correct'].mean()
                                
                                logger.info(f"Original Model - Upset Accuracy: {orig_upset_acc:.4f}")
                                logger.info(f"Enhanced Model - Upset Accuracy: {enhanced_upset_acc:.4f}")
                                
                                upset_diff = enhanced_upset_acc - orig_upset_acc
                                logger.info(f"Upset Accuracy Improvement: {upset_diff:.4f} ({upset_diff * 100:.2f}%)")
                    else:
                        logger.warning(f"No men's tournament results found for seasons {prediction_seasons}")
                else:
                    logger.warning(f"Men's tournament results file not found: {mens_results_path}")
                
                # Load women's tournament results (similar to men's)
                womens_results_path = os.path.join(data_dir, 'WNCAATourneyDetailedResults.csv')
                if os.path.exists(womens_results_path):
                    womens_results = pd.read_csv(womens_results_path)
                    womens_results = womens_results[womens_results['Season'].isin(prediction_seasons)]
                    
                    if not womens_results.empty:
                        logger.info(f"Found {len(womens_results)} women's tournament games for seasons {prediction_seasons}")
                        
                        # Evaluate original model
                        logger.info("Evaluating original women's model predictions...")
                        w_original_evaluation = evaluate_predictions_against_actual(
                            womens_original_preds, 
                            womens_results,
                            gender="women's"
                        )
                        
                        # Evaluate enhanced model
                        logger.info("Evaluating enhanced women's model predictions...")
                        w_enhanced_evaluation = evaluate_predictions_against_actual(
                            womens_enhanced_df,
                            womens_results,
                            gender="women's"
                        )
                        
                        # Compare results
                        logger.info("\n===== COMPARISON: ORIGINAL vs ENHANCED WOMEN'S MODEL =====")
                        logger.info(f"Original Model - Overall Accuracy: {w_original_evaluation['accuracy']:.4f}")
                        logger.info(f"Enhanced Model - Overall Accuracy: {w_enhanced_evaluation['accuracy']:.4f}")
                        
                        w_accuracy_diff = w_enhanced_evaluation['accuracy'] - w_original_evaluation['accuracy']
                        logger.info(f"Accuracy Improvement: {w_accuracy_diff:.4f} ({w_accuracy_diff * 100:.2f}%)")
                        
                        logger.info(f"Original Model - Brier Score: {w_original_evaluation['brier_score']:.4f}")
                        logger.info(f"Enhanced Model - Brier Score: {w_enhanced_evaluation['brier_score']:.4f}")
                        
                        w_brier_diff = w_original_evaluation['brier_score'] - w_enhanced_evaluation['brier_score']
                        logger.info(f"Brier Score Improvement: {w_brier_diff:.4f} ({w_brier_diff / w_original_evaluation['brier_score'] * 100:.2f}%)")
                    else:
                        logger.warning(f"No women's tournament results found for seasons {prediction_seasons}")
                else:
                    logger.warning(f"Women's tournament results file not found: {womens_results_path}")
    else:
        logger.error("Enhanced models not available. Cannot generate predictions.")
    
    # Report completion and timing
    end_time = datetime.now()
    elapsed = end_time - start_time
    logger.info(f"Process completed in {elapsed}")
    logger.info(f"Enhanced models and predictions are ready in the {models_dir} directory")
    logger.info(f"Use the enhanced submission file for significantly improved results!")

if __name__ == "__main__":
    main()