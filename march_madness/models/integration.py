import pandas as pd
import numpy as np
import pickle
import os
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier

from ..models.round_specific import (train_round_specific_models, 
                                  create_enhanced_upset_features,
                                  predict_with_round_specific_models)
from ..models.common import create_feature_interactions, drop_redundant_features
from ..utils.data_access import get_data_with_index

# Set up logging
logger = logging.getLogger(__name__)

def train_enhanced_model(X_train, y_train, gender="men's", random_state=42):
    """
    Train an enhanced model with round-specific components for better performance
    in NCAA tournament predictions
    
    Args:
        X_train: Training features
        y_train: Target values
        gender: 'men's' or 'women's' tournament
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary containing main ensemble model and round-specific models
    """
    print(f"Training enhanced {gender} tournament model...")
    
    # 1. Add enhanced upset detection features
    X_enhanced = create_enhanced_upset_features(X_train, gender=gender)
    print(f"Added {len(X_enhanced.columns) - len(X_train.columns)} upset detection features")
    
    # 2. Create feature interactions
    X_interactions = create_feature_interactions(X_enhanced)
    print(f"Created feature interactions, shape: {X_interactions.shape}")
    
    # 3. Drop redundant features
    X_reduced, dropped_features = drop_redundant_features(X_interactions, threshold=0.95)
    print(f"Removed {len(dropped_features)} redundant features, remaining: {X_reduced.shape[1]}")
    
    # 4. Create main ensemble model
    main_model = create_gender_specific_ensemble(gender, random_state)
    
    # 5. Fit the main model
    main_model.fit(X_reduced, y_train)
    print("Trained main ensemble model")
    
    # 6. Train round-specific models if round info is available
    round_models = None
    if 'ExpectedRound' in X_train.columns:
        tournament_rounds = X_train['ExpectedRound'].unique()
        round_models = train_round_specific_models(X_train, y_train, tournament_rounds)
        print(f"Trained {len(round_models)} round-specific models")
    
    # 7. Return both main model and round-specific models
    return {
        'main_model': main_model,
        'round_models': round_models,
        'feature_names': X_reduced.columns.tolist(),
        'dropped_features': dropped_features
    }

def create_gender_specific_ensemble(gender, random_state=42):
    """
    Create an optimized ensemble model based on gender-specific tournament characteristics
    
    Args:
        gender: 'men's' or 'women's' tournament
        random_state: Random seed for reproducibility
        
    Returns:
        VotingClassifier ensemble model
    """
    if gender == "men's":
        # Men's tournament model - optimized for upset detection and late rounds
        xgb_model = XGBClassifier(
            n_estimators=800,
            learning_rate=0.02,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.75,
            gamma=0.1,
            reg_alpha=0.2,
            reg_lambda=1.2,
            min_child_weight=3,
            scale_pos_weight=1.8,  # Increased to better detect upsets
            random_state=random_state
        )
        
        lgb_model = LGBMClassifier(
            n_estimators=800,
            learning_rate=0.02,
            max_depth=6,
            num_leaves=32,
            subsample=0.8,
            colsample_bytree=0.75,
            reg_alpha=0.2,
            reg_lambda=1.0,
            min_child_samples=15,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1
        )
        
        rf_model = RandomForestClassifier(
            n_estimators=600,
            max_depth=10,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced_subsample',
            random_state=random_state,
            n_jobs=-1
        )
        
        # Create ensemble with optimized weights
        ensemble = VotingClassifier(
            estimators=[
                ('xgb', xgb_model),
                ('lgb', lgb_model),
                ('rf', rf_model)
            ],
            voting='soft',
            weights=[0.45, 0.35, 0.2]  # Higher weight to XGBoost which handles upsets better
        )
        
    else:
        # Women's tournament model - more emphasis on seed strength and favorites
        xgb_model = XGBClassifier(
            n_estimators=600,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.85,
            colsample_bytree=0.8,
            gamma=0.1,
            reg_alpha=0.15,
            reg_lambda=1.0,
            min_child_weight=2,
            scale_pos_weight=1.5,
            random_state=random_state
        )
        
        lgb_model = LGBMClassifier(
            n_estimators=600,
            learning_rate=0.03,
            max_depth=7,
            num_leaves=48,
            subsample=0.85,
            colsample_bytree=0.8,
            reg_alpha=0.15,
            reg_lambda=1.0,
            min_child_samples=10,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1
        )
        
        rf_model = RandomForestClassifier(
            n_estimators=500,
            max_depth=12,
            min_samples_split=3,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1
        )
        
        # Create ensemble with weights for women's tournament
        ensemble = VotingClassifier(
            estimators=[
                ('xgb', xgb_model),
                ('lgb', lgb_model),
                ('rf', rf_model)
            ],
            voting='soft',
            weights=[0.4, 0.4, 0.2]  # More balanced between XGBoost and LightGBM
        )
    
    return ensemble

def predict_with_enhanced_model(model_dict, X_test, gender="men's"):
    """
    Generate predictions using the enhanced model, including round-specific models
    and proper calibration
    
    Args:
        model_dict: Dictionary containing main model and round-specific models
        X_test: Test features
        gender: 'men's' or 'women's' tournament
        
    Returns:
        Array of calibrated predictions
    """
    print(f"Generating {gender} tournament predictions with enhanced model...")
    
    # Extract components from model dictionary
    main_model = model_dict['main_model']
    round_models = model_dict['round_models']
    
    # 1. Add enhanced upset detection features
    X_enhanced = create_enhanced_upset_features(X_test, gender=gender)
    
    # 2. Create feature interactions
    X_interactions = create_feature_interactions(X_enhanced)
    
    # 3. Ensure X_test has the right features for the main model
    if hasattr(main_model, 'feature_names_in_'):
        main_features = main_model.feature_names_in_
        
        # Check if we need to adjust features
        missing_features = [f for f in main_features if f not in X_interactions.columns]
        extra_features = [f for f in X_interactions.columns if f not in main_features]
        
        if missing_features:
            print(f"Warning: {len(missing_features)} features missing from test data")
            
            # Add missing features with zeros
            for feature in missing_features:
                X_interactions[feature] = 0
                
        # Select only the features used by the main model
        X_main = X_interactions[main_features]
    else:
        # If no feature_names_in_ attribute, just use what we have
        X_main = X_interactions
    
    # 4. Get main model predictions
    main_preds = main_model.predict_proba(X_main)[:, 1]
    
    # 5. If we have round models and round information, use them
    if round_models and 'ExpectedRound' in X_test.columns:
        # Get round-specific predictions
        round_preds = predict_with_round_specific_models(X_test, round_models)
        
        # Blend main and round-specific predictions
        # Weight more toward round-specific models for later rounds
        blend_weights = {
            'Championship': 0.7,  # 70% round-specific for championship games
            'Final4': 0.7,        # 70% round-specific for Final Four
            'Elite8': 0.6,        # 60% round-specific for Elite Eight
            'Sweet16': 0.6,       # 60% round-specific for Sweet Sixteen
            'Round32': 0.5,       # 50% round-specific for Round of 32
            'Round64': 0.4        # 40% round-specific for Round of 64
        }
        
        # Initialize blended predictions as a copy of main predictions
        blended_preds = np.copy(main_preds)
        
        # Apply round-specific blending
        for i, round_name in enumerate(X_test['ExpectedRound']):
            if round_name in blend_weights:
                weight = blend_weights[round_name]
            else:
                weight = 0.5  # Default weight
            
            blended_preds[i] = (weight * round_preds[i]) + ((1 - weight) * main_preds[i])
        
        return blended_preds
    else:
        # If no round models or round information, just use main predictions
        return main_preds

def save_enhanced_model(model_dict, feature_cols, filename):
    """
    Save the enhanced model to disk
    
    Args:
        model_dict: Dictionary containing main model and round-specific models
        feature_cols: Feature columns used by the model
        filename: Output filename
    """
    # Create a dictionary with all necessary components
    save_dict = {
        'model': model_dict,
        'feature_cols': feature_cols,
        'version': '2.0'  # Version tracking
    }
    
    # Save to disk
    with open(filename, 'wb') as f:
        pickle.dump(save_dict, f)
    
    print(f"Enhanced model saved to {filename}")

def load_enhanced_model(filename):
    """
    Load an enhanced model from disk
    
    Args:
        filename: Input filename
        
    Returns:
        Tuple of (model_dict, feature_cols)
    """
    with open(filename, 'rb') as f:
        loaded = pickle.load(f)
    
    # Check if this is an enhanced model
    if isinstance(loaded, dict) and 'version' in loaded and loaded['version'] == '2.0':
        return loaded['model'], loaded['feature_cols']
    else:
        # Legacy model format
        print("Warning: Loading legacy model format. Some enhanced features may not be available.")
        return loaded, None

def enhance_existing_models(mens_model, mens_metadata, 
                          womens_model, womens_metadata,
                          mens_data, womens_data, 
                          output_dir='models'):
    """
    Enhance your existing models with the new specialized components
    
    Args:
        mens_model: Existing men's model
        mens_metadata: Existing men's metadata
        womens_model: Existing women's model
        womens_metadata: Existing women's metadata
        mens_data: Men's modeling data dictionary
        womens_data: Women's modeling data dictionary
        output_dir: Directory to save enhanced models
        
    Returns:
        Dictionary with enhanced models and metadata
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    enhanced_models = {}
    
    # Process men's model
    print("\n===== Enhancing men's tournament model =====")
    
    # Debug: Check which seasons are available in season_matchups
    seasons_avail = list(mens_data.get('season_matchups', {}).keys())
    print(f"Available seasons in men's season_matchups: {sorted(seasons_avail)}")
    
    # Get all seasons up to 2022 (training + validation)
    train_seasons = [s for s in seasons_avail if s <= 2022]
    print(f"Using training seasons: {sorted(train_seasons)}")
    
    # Get tournament data
    tourney_data = mens_data.get('tourney_data', pd.DataFrame())
    train_tourney = tourney_data[tourney_data['Season'].isin(train_seasons)]
    print(f"Found {len(train_tourney)} men's tournament games for training seasons")
    
    # Create train data for men's model using 2015-2022 data
    # Approach 1: Use existing prediction data that contains all teams, then match to actual tournament results
    mens_train_data = []
    mens_targets = []
    total_matchups = 0
    matched_games = 0
    
    # For each training season, get the matchups and match to tournament results
    for season in sorted(train_seasons):
        season_matchups = mens_data['season_matchups'][season]
        total_matchups += len(season_matchups)
        
        # Get tournament results for this season
        season_tourney = train_tourney[train_tourney['Season'] == season]
        print(f"Season {season}: {len(season_matchups)} matchups, {len(season_tourney)} tournament games")
        
        # For each tournament game, find the corresponding matchup(s)
        season_targets = []
        
        for _, game in season_tourney.iterrows():
            # Create matchup ID - ensure consistent format
            w_team = game['WTeamID']
            l_team = game['LTeamID']
            matchup_id = f"{season}_{min(w_team, l_team)}_{max(w_team, l_team)}"
            
            # Find all matchups with this ID
            matchup_rows = season_matchups[season_matchups['MatchupID'] == matchup_id]
            
            if len(matchup_rows) == 0:
                logger.warning(f"No matching matchup found for game {matchup_id}")
                continue
                
            # For each matching matchup, determine the target
            for idx, row in matchup_rows.iterrows():
                if row['Team1ID'] == w_team and row['Team2ID'] == l_team:
                    # Team1 won
                    season_targets.append((idx, 1))
                    matched_games += 1
                elif row['Team1ID'] == l_team and row['Team2ID'] == w_team:
                    # Team1 lost
                    season_targets.append((idx, 0))
                    matched_games += 1
        
        # Only add seasons that have matched games
        if season_targets:
            mens_train_data.append(season_matchups)
            mens_targets.extend(season_targets)
            print(f"  Found {len(season_targets)} matched games for season {season}")
        else:
            print(f"  WARNING: No matched games found for season {season}")
    
    print(f"Total matchups: {total_matchups}, Matched games: {matched_games}")
    
    # If we have enough training data, create the model
    if mens_train_data and len(mens_targets) >= 50:  # Reasonable minimum for training
        # Combine all training data
        mens_train_df = pd.concat(mens_train_data, ignore_index=True)
        
        # Apply targets to training data
        mens_train_df['Target'] = 0
        for idx, target in mens_targets:
            # Find the index in the concatenated dataframe
            try:
                new_idx = mens_train_df.index[mens_train_df['MatchupID'] == mens_train_df.loc[idx, 'MatchupID']]
                if len(new_idx) > 0:
                    mens_train_df.loc[new_idx[0], 'Target'] = target
            except (KeyError, IndexError):
                # Index not found, try to match by MatchupID
                match_rows = mens_train_df[mens_train_df['MatchupID'] == mens_train_df.loc[idx, 'MatchupID']]
                if len(match_rows) > 0:
                    mens_train_df.loc[match_rows.index[0], 'Target'] = target
        
        # Filter to only include rows with targets
        mens_train_df = mens_train_df[mens_train_df['Target'].isin([0, 1])]
        
        if len(mens_train_df) > 0:
            print(f"Created men's training dataset with {len(mens_train_df)} samples")
            
            # Debug: Check target distribution
            print(f"Target distribution: {mens_train_df['Target'].value_counts().to_dict()}")
            
            # Prepare features and target
            X_mens = mens_train_df.drop(['Season', 'Team1ID', 'Team2ID', 'Target', 'MatchupID'], 
                                     axis=1, errors='ignore')
            if 'Pred' in X_mens.columns:
                X_mens.drop('Pred', axis=1, inplace=True, errors='ignore')
                
            y_mens = mens_train_df['Target']
            
            # Check for duplicate columns
            dupes = X_mens.columns[X_mens.columns.duplicated()].tolist()
            if dupes:
                print(f"WARNING: Found duplicate columns: {dupes}")
                X_mens = X_mens.loc[:, ~X_mens.columns.duplicated()]
            
            # Train enhanced model
            mens_enhanced = train_enhanced_model(X_mens, y_mens, gender="men's")
            
            # Save enhanced model
            enhanced_models['mens_model'] = mens_enhanced
            enhanced_models['mens_feature_cols'] = X_mens.columns.tolist()
            
            # Save to disk
            save_enhanced_model(mens_enhanced, X_mens.columns.tolist(), 
                               os.path.join(output_dir, 'mens_enhanced_model.pkl'))
            print(f"Enhanced men's model saved to {os.path.join(output_dir, 'mens_enhanced_model.pkl')}")
        else:
            print("Warning: No men's training data with targets found after filtering")
    else:
        print("Warning: Insufficient men's training data found")
    
    # Process women's model (similar to men's)
    print("\n===== Enhancing women's tournament model =====")
    
    # Check which seasons are available in season_matchups
    w_seasons_avail = list(womens_data.get('season_matchups', {}).keys())
    print(f"Available seasons in women's season_matchups: {sorted(w_seasons_avail)}")
    
    # Get all seasons up to 2022 (training + validation)
    w_train_seasons = [s for s in w_seasons_avail if s <= 2022]
    print(f"Using training seasons: {sorted(w_train_seasons)}")
    
    # Get tournament data
    w_tourney_data = womens_data.get('tourney_data', pd.DataFrame())
    w_train_tourney = w_tourney_data[w_tourney_data['Season'].isin(w_train_seasons)]
    print(f"Found {len(w_train_tourney)} women's tournament games for training seasons")
    
    # Create train data for women's model using 2015-2022 data
    womens_train_data = []
    womens_targets = []
    w_total_matchups = 0
    w_matched_games = 0
    
    # For each training season, get the matchups and match to tournament results
    for season in sorted(w_train_seasons):
        try:
            season_matchups = womens_data['season_matchups'][season]
            w_total_matchups += len(season_matchups)
            
            # Get tournament results for this season
            season_tourney = w_train_tourney[w_train_tourney['Season'] == season]
            print(f"Season {season}: {len(season_matchups)} matchups, {len(season_tourney)} tournament games")
            
            # For each tournament game, find the corresponding matchup(s)
            season_targets = []
            
            for _, game in season_tourney.iterrows():
                # Create matchup ID - ensure consistent format
                w_team = game['WTeamID']
                l_team = game['LTeamID']
                matchup_id = f"{season}_{min(w_team, l_team)}_{max(w_team, l_team)}"
                
                # Find all matchups with this ID
                matchup_rows = season_matchups[season_matchups['MatchupID'] == matchup_id]
                
                if len(matchup_rows) == 0:
                    logger.warning(f"No matching matchup found for women's game {matchup_id}")
                    continue
                    
                # For each matching matchup, determine the target
                for idx, row in matchup_rows.iterrows():
                    if row['Team1ID'] == w_team and row['Team2ID'] == l_team:
                        # Team1 won
                        season_targets.append((idx, 1))
                        w_matched_games += 1
                    elif row['Team1ID'] == l_team and row['Team2ID'] == w_team:
                        # Team1 lost
                        season_targets.append((idx, 0))
                        w_matched_games += 1
            
            # Only add seasons that have matched games
            if season_targets:
                womens_train_data.append(season_matchups)
                womens_targets.extend(season_targets)
                print(f"  Found {len(season_targets)} matched games for season {season}")
            else:
                print(f"  WARNING: No matched games found for season {season}")
        except Exception as e:
            print(f"Error processing women's season {season}: {str(e)}")
    
    print(f"Total matchups: {w_total_matchups}, Matched games: {w_matched_games}")
    
    # If we have enough training data, create the model
    if womens_train_data and len(womens_targets) >= 50:  # Reasonable minimum for training
        # Combine all training data
        womens_train_df = pd.concat(womens_train_data, ignore_index=True)
        
        # Apply targets to training data
        womens_train_df['Target'] = 0
        for idx, target in womens_targets:
            # Find the index in the concatenated dataframe
            try:
                new_idx = womens_train_df.index[womens_train_df['MatchupID'] == womens_train_df.loc[idx, 'MatchupID']]
                if len(new_idx) > 0:
                    womens_train_df.loc[new_idx[0], 'Target'] = target
            except (KeyError, IndexError):
                # Index not found, try to match by MatchupID
                match_rows = womens_train_df[womens_train_df['MatchupID'] == womens_train_df.loc[idx, 'MatchupID']]
                if len(match_rows) > 0:
                    womens_train_df.loc[match_rows.index[0], 'Target'] = target
        
        # Filter to only include rows with targets
        womens_train_df = womens_train_df[womens_train_df['Target'].isin([0, 1])]
        
        if len(womens_train_df) > 0:
            print(f"Created women's training dataset with {len(womens_train_df)} samples")
            
            # Debug: Check target distribution
            print(f"Target distribution: {womens_train_df['Target'].value_counts().to_dict()}")
            
            # Prepare features and target
            X_womens = womens_train_df.drop(['Season', 'Team1ID', 'Team2ID', 'Target', 'MatchupID'], 
                                         axis=1, errors='ignore')
            if 'Pred' in X_womens.columns:
                X_womens.drop('Pred', axis=1, inplace=True, errors='ignore')
                
            y_womens = womens_train_df['Target']
            
            # Check for duplicate columns
            dupes = X_womens.columns[X_womens.columns.duplicated()].tolist()
            if dupes:
                print(f"WARNING: Found duplicate columns: {dupes}")
                X_womens = X_womens.loc[:, ~X_womens.columns.duplicated()]
            
            # Train enhanced model
            womens_enhanced = train_enhanced_model(X_womens, y_womens, gender="women's")
            
            # Save enhanced model
            enhanced_models['womens_model'] = womens_enhanced
            enhanced_models['womens_feature_cols'] = X_womens.columns.tolist()
            
            # Save to disk
            save_enhanced_model(womens_enhanced, X_womens.columns.tolist(),
                               os.path.join(output_dir, 'womens_enhanced_model.pkl'))
            print(f"Enhanced women's model saved to {os.path.join(output_dir, 'womens_enhanced_model.pkl')}")
        else:
            print("Warning: No women's training data with targets found after filtering")
    else:
        print("Warning: Insufficient women's training data found")
    
    return enhanced_models