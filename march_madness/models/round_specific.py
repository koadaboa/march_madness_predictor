import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
from scipy.special import expit  # For logistic function
from ..utils.data_access import get_data_with_index

def train_round_specific_models(X_train, y_train, tournament_rounds, round_column='ExpectedRound'):
    """
    Train specialized models for each tournament round with optimized hyperparameters 
    and dynamic feature selection
    
    Args:
        X_train: Training features
        y_train: Target values
        tournament_rounds: List of tournament rounds
        round_column: Column name containing round information
        
    Returns:
        Dictionary of round-specific models
    """
    print(f"Training specialized models for {len(tournament_rounds)} tournament rounds...")
    round_models = {}
    
    # Create standard scaler for all data
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train.drop(round_column, axis=1, errors='ignore')),
        columns=X_train.drop(round_column, axis=1, errors='ignore').columns
    )
    
    # Define a base set of features that are always important
    base_features = [
        'SeedDiff', 'Team1Seed', 'Team2Seed',
        'WinRateDiff', 'OffEfficiencyDiff', 'DefEfficiencyDiff', 
        'SOSPercentileDiff', 'TourneyAppearancesDiff'
    ]
    
    # Define key rounds that need specialized handling
    critical_rounds = ['Sweet16', 'Elite8', 'Final4', 'Championship']
    early_rounds = ['Round64', 'Round32']
    
    # Define round-specific feature importances based on domain knowledge and analysis
    round_feature_emphasis = {
        'Championship': {
            'emphasize': ['Team1PressureScore', 'Team2PressureScore', 'CoachTourneyExpDiff', 
                         'Team1Championships', 'Team2Championships', 'CoachChampionshipsDiff',
                         'DefEfficiencyDiff', 'CloseGameWinRateDiff'],
            'weight': 2.0
        },
        'Final4': {
            'emphasize': ['Team1MomentumChange', 'Team2MomentumChange', 'RoundWinRateDiff',
                         'DefEfficiencyDiff', 'Team1FG3Pct', 'Team2FG3Pct',
                         'MomentumChangeDiff', 'UnderdogExperienceFactor'],
            'weight': 2.0
        },
        'Elite8': {
            'emphasize': ['Team1PressureScore', 'Team2PressureScore', 'PressureScoreDiff',
                         'Team1TourneyWinRate', 'Team2TourneyWinRate',
                         'Team1DefEfficiency', 'Team2DefEfficiency'],
            'weight': 1.5
        },
        'Sweet16': {
            'emphasize': ['PressureScoreDiff', 'ScoreMarginLast5_Diff', 'DefEfficiencyDiff',
                         'Team1FG3Pct', 'Team2FG3Pct', 'UnderdogMomentum'],
            'weight': 1.5
        },
        'Round32': {
            'emphasize': ['AvgRoundPerformanceDiff', 'Team1OffEfficiency', 'Team2OffEfficiency',
                         'SOSDiff', 'Team1SOS', 'Team2SOS'],
            'weight': 1.2
        },
        'Round64': {
            'emphasize': ['SeedDiff', 'SOSDiff', 'ThreePointUpsetFactor', 'UnderdogDefenseFactor'],
            'weight': 1.0
        }
    }
    
    # Round-specific hyperparameters
    round_hyperparams = {
        'Championship': {
            'xgb': {'n_estimators': 600, 'learning_rate': 0.02, 'max_depth': 4, 'subsample': 0.85, 
                   'colsample_bytree': 0.8, 'scale_pos_weight': 1.5},
            'lgb': {'n_estimators': 600, 'learning_rate': 0.02, 'max_depth': 4, 'num_leaves': 20, 
                   'subsample': 0.8, 'colsample_bytree': 0.8, 'class_weight': 'balanced'},
            'rf': {'n_estimators': 600, 'max_depth': 10, 'min_samples_split': 4, 'min_samples_leaf': 2,
                  'class_weight': 'balanced_subsample'}
        },
        'Final4': {
            'xgb': {'n_estimators': 600, 'learning_rate': 0.02, 'max_depth': 4, 'subsample': 0.7, 
                   'colsample_bytree': 0.7, 'scale_pos_weight': 2.0},  # More weight to minority class
            'lgb': {'n_estimators': 600, 'learning_rate': 0.02, 'max_depth': 4, 'num_leaves': 16, 
                   'subsample': 0.7, 'colsample_bytree': 0.7, 'class_weight': 'balanced'},
            'rf': {'n_estimators': 600, 'max_depth': 8, 'min_samples_split': 5, 'min_samples_leaf': 2,
                  'class_weight': 'balanced_subsample'}
        },
        'Elite8': {
            'xgb': {'n_estimators': 500, 'learning_rate': 0.03, 'max_depth': 4, 'subsample': 0.8, 
                   'colsample_bytree': 0.8, 'scale_pos_weight': 1.8},
            'lgb': {'n_estimators': 500, 'learning_rate': 0.03, 'max_depth': 5, 'num_leaves': 24, 
                   'subsample': 0.8, 'colsample_bytree': 0.8, 'class_weight': 'balanced'},
            'rf': {'n_estimators': 500, 'max_depth': 9, 'min_samples_split': 4, 'min_samples_leaf': 2,
                  'class_weight': 'balanced'}
        },
        'Sweet16': {
            'xgb': {'n_estimators': 500, 'learning_rate': 0.03, 'max_depth': 5, 'subsample': 0.8, 
                   'colsample_bytree': 0.8, 'scale_pos_weight': 2.0},  # Sweet16 has more upsets
            'lgb': {'n_estimators': 500, 'learning_rate': 0.03, 'max_depth': 6, 'num_leaves': 32, 
                   'subsample': 0.8, 'colsample_bytree': 0.8, 'class_weight': 'balanced'},
            'rf': {'n_estimators': 500, 'max_depth': 10, 'min_samples_split': 3, 'min_samples_leaf': 2,
                  'class_weight': 'balanced_subsample'}
        },
        'Round32': {
            'xgb': {'n_estimators': 400, 'learning_rate': 0.04, 'max_depth': 5, 'subsample': 0.9, 
                   'colsample_bytree': 0.9, 'scale_pos_weight': 1.5},
            'lgb': {'n_estimators': 400, 'learning_rate': 0.04, 'max_depth': 6, 'num_leaves': 40, 
                   'subsample': 0.9, 'colsample_bytree': 0.9, 'class_weight': 'balanced'},
            'rf': {'n_estimators': 400, 'max_depth': 12, 'min_samples_split': 3, 'min_samples_leaf': 2,
                  'class_weight': 'balanced_subsample'}
        },
        'Round64': {
            'xgb': {'n_estimators': 400, 'learning_rate': 0.05, 'max_depth': 6, 'subsample': 0.9, 
                   'colsample_bytree': 0.9, 'scale_pos_weight': 1.2},
            'lgb': {'n_estimators': 400, 'learning_rate': 0.05, 'max_depth': 7, 'num_leaves': 48, 
                   'subsample': 0.9, 'colsample_bytree': 0.9, 'class_weight': 'balanced'},
            'rf': {'n_estimators': 400, 'max_depth': 15, 'min_samples_split': 2, 'min_samples_leaf': 1,
                  'class_weight': 'balanced_subsample'}
        }
    }
    
    # Configure ensemble weights based on rounds
    ensemble_weights = {
        'Championship': {'xgb': 0.4, 'lgb': 0.3, 'rf': 0.3},
        'Final4': {'xgb': 0.45, 'lgb': 0.35, 'rf': 0.2},  # XGBoost works better for upsets
        'Elite8': {'xgb': 0.4, 'lgb': 0.35, 'rf': 0.25},
        'Sweet16': {'xgb': 0.4, 'lgb': 0.35, 'rf': 0.25},
        'Round32': {'xgb': 0.35, 'lgb': 0.35, 'rf': 0.3},
        'Round64': {'xgb': 0.3, 'lgb': 0.3, 'rf': 0.4},  # RF works better for favorites
    }
    
    # For each round, train a specialized model
    for round_name in tournament_rounds:
        print(f"Training model for {round_name}...")
        
        # Filter data for this round
        round_mask = X_train[round_column] == round_name
        if sum(round_mask) < 10:  # Need minimum examples
            print(f"  Insufficient data for {round_name}, using general model")
            continue
            
        X_round = X_train_scaled[round_mask]
        y_round = y_train[round_mask]
        
        # Apply feature weights for this round
        feature_weights = np.ones(X_round.shape[1])
        
        # Emphasize round-specific features
        if round_name in round_feature_emphasis:
            emphasis_features = round_feature_emphasis[round_name]['emphasize']
            emphasis_weight = round_feature_emphasis[round_name]['weight']
            
            for i, feature in enumerate(X_train_scaled.columns):
                # Apply higher weight to emphasized features
                if any(emph_feat in feature for emph_feat in emphasis_features):
                    feature_weights[i] = emphasis_weight
                # Also ensure base features have at least some emphasis
                elif any(base_feat in feature for base_feat in base_features):
                    feature_weights[i] = max(feature_weights[i], 1.2)
        
        # Apply weighted features to the data
        X_round_weighted = X_round * feature_weights
        
        # Get round-specific hyperparameters (default to Championship if not found)
        round_params = round_hyperparams.get(round_name, round_hyperparams['Championship'])
        round_weights = ensemble_weights.get(round_name, {'xgb': 0.4, 'lgb': 0.3, 'rf': 0.3})
        
        # Train individual models
        xgb_model = XGBClassifier(random_state=42, **round_params['xgb'])
        lgb_model = LGBMClassifier(random_state=42, **round_params['lgb'])
        rf_model = RandomForestClassifier(random_state=42, **round_params['rf'])
        
        # Fit models
        xgb_model.fit(X_round_weighted, y_round)
        lgb_model.fit(X_round_weighted, y_round)
        rf_model.fit(X_round_weighted, y_round)
        
        # Store models in a dictionary
        round_models[round_name] = {
            'models': {
                'xgb': xgb_model,
                'lgb': lgb_model,
                'rf': rf_model
            },
            'weights': round_weights,
            'feature_weights': feature_weights,
            'features': X_train_scaled.columns.tolist(),
            'scaler': scaler
        }
        
        print(f"  Trained {round_name} model with {len(X_round)} samples")
        
    return round_models

def create_enhanced_upset_features(X, gender="men's"):
    """
    Create enhanced features specifically for detecting upsets
    
    Args:
        X: Feature DataFrame
        gender: 'men's' or 'women's' gender
        
    Returns:
        DataFrame with enhanced upset features
    """
    X_enhanced = X.copy()
    
    # Create seed-specific upset indicators based on historical patterns
    if 'Team1Seed' in X.columns and 'Team2Seed' in X.columns:
        # For men's tournament, specific seed matchups have higher upset rates
        if gender == "men's":
            # 5-12 matchups have historically high upset rates (~35%)
            X_enhanced['Upset_5v12'] = (
                ((X['Team1Seed'] == 5) & (X['Team2Seed'] == 12)) | 
                ((X['Team1Seed'] == 12) & (X['Team2Seed'] == 5))
            ).astype(int)
            
            # 6-11 matchups also have high upset rates (~33%)
            X_enhanced['Upset_6v11'] = (
                ((X['Team1Seed'] == 6) & (X['Team2Seed'] == 11)) | 
                ((X['Team1Seed'] == 11) & (X['Team2Seed'] == 6))
            ).astype(int)
            
            # 7-10 matchups tend toward upsets (~40%)
            X_enhanced['Upset_7v10'] = (
                ((X['Team1Seed'] == 7) & (X['Team2Seed'] == 10)) | 
                ((X['Team1Seed'] == 10) & (X['Team2Seed'] == 7))
            ).astype(int)
            
            # 8-9 matchups are effectively toss-ups
            X_enhanced['Upset_8v9'] = (
                ((X['Team1Seed'] == 8) & (X['Team2Seed'] == 9)) | 
                ((X['Team1Seed'] == 9) & (X['Team2Seed'] == 8))
            ).astype(int)
        
        # Women's tournament has different upset patterns
        else:
            # Women's tournament has fewer 5-12 upsets
            X_enhanced['Upset_5v12'] = (
                ((X['Team1Seed'] == 5) & (X['Team2Seed'] == 12)) | 
                ((X['Team1Seed'] == 12) & (X['Team2Seed'] == 5))
            ).astype(int)
            
            # 6-11 and 7-10 upsets are more common in recent seasons
            X_enhanced['Upset_6v11_7v10'] = (
                ((X['Team1Seed'] == 6) & (X['Team2Seed'] == 11)) | 
                ((X['Team1Seed'] == 11) & (X['Team2Seed'] == 6)) |
                ((X['Team1Seed'] == 7) & (X['Team2Seed'] == 10)) | 
                ((X['Team1Seed'] == 10) & (X['Team2Seed'] == 7))
            ).astype(int)
    
    # Create momentum-based upset indicators
    if 'Team1WinRate_Last5' in X.columns and 'Team2WinRate_Last5' in X.columns and 'SeedDiff' in X.columns:
        # Calculate momentum advantage for underdogs
        X_enhanced['UnderdogMomentum'] = np.where(
            X['SeedDiff'] > 0,  # Team1 is underdog (higher seed number)
            X['Team1WinRate_Last5'] - X['Team2WinRate_Last5'],
            X['Team2WinRate_Last5'] - X['Team1WinRate_Last5']
        )
        
        # Create upset potential score based on underdog momentum and seed difference
        X_enhanced['UpsetPotential'] = X_enhanced['UnderdogMomentum'] * np.log1p(abs(X['SeedDiff']))
    
    # Create defense-based upset indicators
    # Strong defensive teams with weaker seeds often cause upsets
    if 'Team1DefEfficiency' in X.columns and 'Team2DefEfficiency' in X.columns and 'SeedDiff' in X.columns:
        # Calculate defensive advantage for underdogs (lower is better for defense)
        X_enhanced['UnderdogDefenseAdvantage'] = np.where(
            X['SeedDiff'] > 0,  # Team1 is underdog
            X['Team2DefEfficiency'] - X['Team1DefEfficiency'], 
            X['Team1DefEfficiency'] - X['Team2DefEfficiency']
        )
        
        # Create defense-based upset potential
        X_enhanced['DefenseUpsetPotential'] = np.where(
            X_enhanced['UnderdogDefenseAdvantage'] > 0,
            X_enhanced['UnderdogDefenseAdvantage'] * np.log1p(abs(X['SeedDiff'])),
            0
        )
    
    # Three-point shooting upset potential
    # Good 3-point shooting underdogs have higher upset potential
    if 'Team1FG3Pct' in X.columns and 'Team2FG3Pct' in X.columns and 'SeedDiff' in X.columns:
        X_enhanced['Underdog3PtAdvantage'] = np.where(
            X['SeedDiff'] > 0,  # Team1 is underdog
            X['Team1FG3Pct'] - X['Team2FG3Pct'],
            X['Team2FG3Pct'] - X['Team1FG3Pct']
        )
        
        # Create 3pt-based upset potential
        X_enhanced['ThreePointUpsetPotential'] = np.where(
            X_enhanced['Underdog3PtAdvantage'] > 0,
            X_enhanced['Underdog3PtAdvantage'] * np.log1p(abs(X['SeedDiff'])),
            0
        )
    
    # Experience-based upset potential
    # Experienced underdogs with tournament history can cause upsets
    if all(col in X.columns for col in ['TourneyAppearancesDiff', 'Team1TourneyAppearances', 
                                        'Team2TourneyAppearances', 'SeedDiff']):
        # Calculate experience advantage for underdogs
        X_enhanced['UnderdogExperienceAdvantage'] = np.where(
            X['SeedDiff'] > 0,  # Team1 is underdog
            np.maximum(0, X['Team1TourneyAppearances'] - X['Team2TourneyAppearances']),
            np.maximum(0, X['Team2TourneyAppearances'] - X['Team1TourneyAppearances'])
        )
        
        # Create experience-based upset potential
        X_enhanced['ExperienceUpsetPotential'] = X_enhanced['UnderdogExperienceAdvantage'] * 0.1
    
    # Conference strength indicators
    # Strong conference underdogs often perform better than their seed
    if all(col in X.columns for col in ['Team1ConfWinsPerTeam', 'Team2ConfWinsPerTeam', 'SeedDiff']):
        # Calculate conference strength advantage for underdogs
        X_enhanced['UnderdogConfAdvantage'] = np.where(
            X['SeedDiff'] > 0,  # Team1 is underdog
            np.maximum(0, X['Team1ConfWinsPerTeam'] - X['Team2ConfWinsPerTeam']),
            np.maximum(0, X['Team2ConfWinsPerTeam'] - X['Team1ConfWinsPerTeam'])
        )
        
        # Create conference-based upset potential
        X_enhanced['ConferenceUpsetPotential'] = X_enhanced['UnderdogConfAdvantage'] * 0.5
    
    # Combine upset potentials into a composite score
    upset_potentials = []
    potential_columns = [
        'UpsetPotential', 'DefenseUpsetPotential', 'ThreePointUpsetPotential',
        'ExperienceUpsetPotential', 'ConferenceUpsetPotential'
    ]
    
    for col in potential_columns:
        if col in X_enhanced.columns:
            upset_potentials.append(X_enhanced[col])
    
    # Create composite score if we have at least one potential metric
    if upset_potentials:
        X_enhanced['CompositeUpsetScore'] = sum(upset_potentials)
    
    # Calculate round-specific upset factors for men's tournament
    if gender == "men's" and 'ExpectedRound' in X.columns:
        # Sweet 16 has different upset dynamics
        sweet16_mask = X['ExpectedRound'] == 'Sweet16'
        if sum(sweet16_mask) > 0:
            if 'CompositeUpsetScore' in X_enhanced.columns:
                # Boost upset score for Sweet 16 games
                X_enhanced.loc[sweet16_mask, 'CompositeUpsetScore'] *= 1.25
                
        # Final Four has very different dynamics
        final4_mask = X['ExpectedRound'] == 'Final4'
        if sum(final4_mask) > 0:
            if 'CompositeUpsetScore' in X_enhanced.columns:
                # Reduce upset score for Final Four (more about team quality)
                X_enhanced.loc[final4_mask, 'CompositeUpsetScore'] *= 0.85
    
    return X_enhanced

def predict_with_round_specific_models(X_test, round_models, round_column='ExpectedRound'):
    """
    Generate predictions using round-specific models with round-appropriate calibration
    
    Args:
        X_test: Test features
        round_models: Dictionary of round-specific models
        round_column: Column name containing round information
        
    Returns:
        Array of calibrated predictions
    """
    # Get the round for each test sample
    if round_column in X_test.columns:
        rounds = X_test[round_column].values
    else:
        # Default to Championship if round not specified
        rounds = np.array(['Championship'] * len(X_test))
    
    # Prepare features (removing round column)
    X_features = X_test.drop(round_column, axis=1, errors='ignore')
    
    # Initialize predictions array
    predictions = np.zeros(len(X_test))
    
    # For each sample, apply the appropriate round-specific model
    for i, sample_round in enumerate(rounds):
        # Get features for this sample
        sample_features = X_features.iloc[i:i+1]
        
        # Use round-specific model if available, otherwise default to first round model
        if sample_round in round_models:
            round_model = round_models[sample_round]
        else:
            # Use first available model as fallback
            sample_round = list(round_models.keys())[0]
            round_model = round_models[sample_round]
        
        # Transform features
        sample_scaled = round_model['scaler'].transform(sample_features)
        
        # Apply feature weights
        sample_weighted = sample_scaled * round_model['feature_weights']
        
        # Get predictions from each model in the ensemble
        model_preds = {}
        for model_name, model in round_model['models'].items():
            # Ensure DataFrame has the right features in the right order
            model_features = pd.DataFrame(sample_weighted, columns=sample_features.columns)
            if hasattr(model, 'feature_names_in_'):
                model_features = model_features[model.feature_names_in_]
                
            # Get prediction
            model_preds[model_name] = model.predict_proba(model_features)[0, 1]
        
        # Combine predictions using round-specific weights
        weights = round_model['weights']
        weighted_pred = sum(model_preds[model] * weights[model] for model in model_preds)
        
        # Store prediction
        predictions[i] = weighted_pred
    
    # Get the round-specific calibration parameters
    return calibrate_predictions(predictions, rounds)

def calibrate_predictions(predictions, rounds):
    """
    Apply round-specific calibration to improve prediction accuracy,
    especially for upsets and high confidence predictions
    
    Args:
        predictions: Raw model predictions
        rounds: Tournament round for each prediction
        
    Returns:
        Calibrated predictions
    """
    # Create a copy of predictions to modify
    calibrated = np.copy(predictions)
    
    # Define round-specific calibration parameters based on analysis
    # These values push predictions toward being more accurate based on historical patterns
    calibration_params = {
        'Championship': {
            'center': 0.5,          # Center probability
            'stretch': 0.9,         # Stretch factor 
            'favorite_boost': 0.02,  # Slightly favor higher seeds
            'upset_threshold': 0.45  # Below this is considered upset territory
        },
        'Final4': {
            'center': 0.5,
            'stretch': 0.85,
            'favorite_boost': 0.00,  # No favorite boost (upsets common)
            'upset_threshold': 0.45
        },
        'Elite8': {
            'center': 0.5,
            'stretch': 0.92,
            'favorite_boost': 0.01,
            'upset_threshold': 0.44
        },
        'Sweet16': {
            'center': 0.5,
            'stretch': 0.94,
            'favorite_boost': -0.02,  # Actually favor upsets slightly
            'upset_threshold': 0.42
        },
        'Round32': {
            'center': 0.5,
            'stretch': 0.96,
            'favorite_boost': 0.03, 
            'upset_threshold': 0.41
        },
        'Round64': {
            'center': 0.5,
            'stretch': 0.98,
            'favorite_boost': 0.05,  # Stronger favorite boost (fewer upsets)
            'upset_threshold': 0.40
        }
    }
    
    # Apply calibration to each prediction
    for i, (pred, round_name) in enumerate(zip(predictions, rounds)):
        # Use appropriate calibration parameters for this round
        if round_name in calibration_params:
            params = calibration_params[round_name]
        else:
            # Default to Championship parameters
            params = calibration_params['Championship']
        
        # Apply calibration
        # 1. Center around 0.5
        centered = pred - params['center']
        
        # 2. Apply stretch factor (pushes predictions away from 0.5)
        stretched = centered * params['stretch']
        
        # 3. Recenter
        recalibrated = stretched + params['center']
        
        # 4. Apply favorite boost/underdog boost based on prediction
        if pred > params['upset_threshold']:
            # Favorite boost
            recalibrated += params['favorite_boost']
        else:
            # Underdog boost (opposite of favorite boost)
            recalibrated -= params['favorite_boost']
        
        # 5. Ensure prediction is in valid range
        calibrated[i] = np.clip(recalibrated, 0.01, 0.99)
    
    return calibrated

# Example usage
if __name__ == "__main__":
    # This will be executed when the module is run directly
    print("Module functions:")
    print("1. train_round_specific_models - Train models specialized for each tournament round")
    print("2. create_enhanced_upset_features - Create features to better detect potential upsets")
    print("3. predict_with_round_specific_models - Make predictions with round-specific models")
    print("4. calibrate_predictions - Apply sophisticated calibration based on historical patterns")