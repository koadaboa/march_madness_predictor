import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss
from ..features.matchup import (calculate_seed_based_probability, create_upset_specific_features)
from ..models.training import (handle_class_imbalance, gender_specific_feature_selection, 
create_mens_specific_model, create_womens_specific_model, train_round_specific_models)
from ..models.evaluation import calibrate_by_expected_round, calibrate_mens_predictions
from ..models.prediction import run_tournament_simulation_pre_tournament
from march_madness.models.common import create_feature_interactions, drop_redundant_features 
import warnings
warnings.filterwarnings("ignore")        

def train_and_predict_model(modeling_data, gender, training_seasons, validation_season, prediction_seasons,
                           model=None, feature_cols=None, scaler=None, dropped_features=None):
    """
    Trains model and generates predictions using prepared data (Steps 10-16)

    Args:
        modeling_data: Dictionary with prepared data from prepare_modeling_data
        gender: 'men' or 'women' to identify which dataset is being processed
        training_seasons: List of seasons to use for training
        validation_season: Season to use for validation
        prediction_seasons: List of seasons to generate predictions for
        model: Pre-trained model (optional, for prediction mode)
        feature_cols: Feature columns used by the model (optional, for prediction mode)
        scaler: Fitted scaler for features (optional, for prediction mode)
        dropped_features: Features that were dropped during training (optional)

    Returns:
        DataFrame with predictions or trained model based on mode
    """
    print(f"\n=== Training and predicting for {gender}'s NCAA Basketball Tournament ===\n")

    # Extract prepared data with error handling
    try:
        season_matchups = modeling_data.get('season_matchups', {})
        if not season_matchups:
            print("ERROR: 'season_matchups' is empty or missing from modeling_data")
            print(f"Available keys: {list(modeling_data.keys())}")
            return pd.DataFrame(columns=['Season', 'Team1ID', 'Team2ID', 'Pred', 'MatchupID'])

        df_tourney = modeling_data.get('tourney_data', pd.DataFrame())
        df_seed = modeling_data.get('df_seed', pd.DataFrame())
        df_teams = modeling_data.get('df_teams', pd.DataFrame())
        df_tourney_slots = modeling_data.get('df_tourney_slots', pd.DataFrame())
    except Exception as e:
        print(f"ERROR extracting data from modeling_data: {str(e)}")
        return pd.DataFrame(columns=['Season', 'Team1ID', 'Team2ID', 'Pred', 'MatchupID'])

    # Determine mode
    is_training_mode = len(training_seasons) > 0 and len(prediction_seasons) == 0
    is_prediction_mode = len(prediction_seasons) > 0

    print(f"Running in {'TRAINING' if is_training_mode else 'PREDICTION'} mode")

    if is_training_mode:
        # Create training dataset
        print("\n10. Preparing data for model training...")

        # Combine training seasons data
        training_matchups_list = []
        for season in training_seasons:
            if season in season_matchups:
                training_matchups_list.append(season_matchups[season])
            else:
                print(f"WARNING: Season {season} not found in season_matchups")

        if not training_matchups_list:
            raise ValueError("No training seasons found in data")

        training_matchups = pd.concat(training_matchups_list, ignore_index=True)

        # Add target column by matching with actual tournament results
        training_with_targets = []

        for season in training_seasons:
            if season not in season_matchups:
                print(f"Skipping season {season} - no matchup data available")
                continue

            season_matchups_df = season_matchups[season]
            season_results = df_tourney[df_tourney['Season'] == season]

            if len(season_results) == 0:
                print(f"WARNING: No tournament results for season {season}")
                continue

            for _, result in season_results.iterrows():
                # Find the matchup in our features dataset
                matchup_id = f"{result['Season']}_{min(result['WTeamID'], result['LTeamID'])}_{max(result['WTeamID'], result['LTeamID'])}"

                # Get all potential matchups that match this actual game
                match_rows = season_matchups_df[season_matchups_df['MatchupID'] == matchup_id]

                # Set target based on who actually won
                for _, row in match_rows.iterrows():
                    row_copy = row.copy()
                    if row['Team1ID'] == result['WTeamID'] and row['Team2ID'] == result['LTeamID']:
                        row_copy['Target'] = 1  # Team1 won
                        training_with_targets.append(pd.DataFrame([row_copy]))
                    elif row['Team1ID'] == result['LTeamID'] and row['Team2ID'] == result['WTeamID']:
                        row_copy['Target'] = 0  # Team1 lost
                        training_with_targets.append(pd.DataFrame([row_copy]))

        if not training_with_targets:
            raise ValueError("No matching games found for training seasons!")
            
        # Combine all training examples with targets
        training_df = pd.concat(training_with_targets, ignore_index=True)
        print(f"Found {len(training_df)} actual game matchups for training")
        
        # Use gender-specific feature selection
        X_train_raw = training_df.drop(['Season', 'Team1ID', 'Team2ID', 'Target', 'ExpectedRound', 'MatchupID'], 
                                    axis=1, errors='ignore')
        y_train = training_df['Target']
        
        feature_cols = gender_specific_feature_selection(X_train_raw, y_train, gender)
        print(f"Selected {len(feature_cols)} gender-specific features for {gender} tournaments")
        
        # Use selected features
        X_train = X_train_raw[feature_cols]

        print("\n11. Creating gender-specific features for better upset detection...")
        X_train_upset = create_upset_specific_features(X_train, gender)
        print(f"Added upset-specific features: {X_train_upset.shape}")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        # Create feature interactions
        print("\n12. Creating feature interactions...")
        X_train_enhanced = create_feature_interactions(X_train)
        print(f"Enhanced features: {X_train_enhanced.shape}")

        # Drop highly correlated features
        print("\n13. Removing redundant features...")
        X_train_reduced, dropped_features = drop_redundant_features(X_train_enhanced, threshold=0.95)
        print(f"Reduced features: {X_train_reduced.shape}, dropped {len(dropped_features)} redundant features")

        # Handle class imbalance
        print("\n14. Handling class imbalance...")
        X_train_balanced, y_train_balanced = handle_class_imbalance(
            X_train_reduced, y_train, method='combined', random_state=42
        )
        print(f"Balanced training data: {X_train_balanced.shape}")

        # Train model
        print(f"\n15. Training gender-specific ensemble model for {gender} tournaments...")
        if gender == "men's":
            model = create_mens_specific_model(random_state=3)
        else:
            model = create_womens_specific_model(random_state=3)
            
        # Fit main ensemble model
        model.fit(X_train_balanced, y_train_balanced)
            
        print("Models trained")

        # Validate model if validation season is provided
        if validation_season:
            print(f"\n16. Validating model on {validation_season} data...")

            if validation_season not in season_matchups:
                print(f"No data available for validation season {validation_season}")
            else:
                # Get validation matchups
                validation_matchups_df = season_matchups[validation_season]
                validation_results = df_tourney[df_tourney['Season'] == validation_season]

                # Match with actual games
                validation_with_targets = []

                for _, result in validation_results.iterrows():
                    matchup_id = f"{result['Season']}_{min(result['WTeamID'], result['LTeamID'])}_{max(result['WTeamID'], result['LTeamID'])}"
                    match_rows = validation_matchups_df[validation_matchups_df['MatchupID'] == matchup_id]

                    for _, row in match_rows.iterrows():
                        row_copy = row.copy()
                        if row['Team1ID'] == result['WTeamID'] and row['Team2ID'] == result['LTeamID']:
                            row_copy['Target'] = 1  # Team1 won
                            validation_with_targets.append(pd.DataFrame([row_copy]))
                        elif row['Team1ID'] == result['LTeamID'] and row['Team2ID'] == result['WTeamID']:
                            row_copy['Target'] = 0  # Team1 lost
                            validation_with_targets.append(pd.DataFrame([row_copy]))

                if validation_with_targets:
                    validation_df = pd.concat(validation_with_targets, ignore_index=True)
                    print(f"Found {len(validation_df)} actual game matchups for validation")

                    # Prepare validation features
                    X_val = validation_df[feature_cols]
                    y_val = validation_df['Target']

                    # Apply transformations
                    X_val_scaled = scaler.transform(X_val)
                    X_val_enhanced = create_feature_interactions(X_val)
                    X_val_reduced = X_val_enhanced.drop(dropped_features, axis=1, errors='ignore')

                    # Make predictions
                    if isinstance(model, dict) and 'main_model' in model and 'round_models' in model:
                        # Using combined model with round-specific models
                        main_model = model['main_model']
                        round_models = model['round_models']
                        
                        # Initialize predictions array
                        val_preds_proba = np.zeros(len(X_val_reduced))
                        
                        # Check if we have round information
                        if 'ExpectedRound' in validation_df.columns:
                            print(f"Using round-specific models for validation where available")
                            
                            # Make predictions for each sample
                            for i, row in validation_df.iterrows():
                                # Get expected round for this matchup
                                round_name = row['ExpectedRound']
                                features = X_val_reduced.loc[i]
                                
                                if round_name in round_models:
                                    # Use round-specific model with weight
                                    round_pred = round_models[round_name].predict_proba(features.values.reshape(1, -1))[0][1]
                                    main_pred = main_model.predict_proba(features.values.reshape(1, -1))[0][1]
                                    
                                    # Blend with more weight to round-specific model in later rounds
                                    if round_name in ['Championship', 'Final4', 'Elite8', 'Sweet16']:
                                        blend_weight = 0.7  # 70% round-specific, 30% main model
                                    else:
                                        blend_weight = 0.5  # 50-50 blend for earlier rounds
                                        
                                    val_preds_proba[i] = (blend_weight * round_pred) + ((1-blend_weight) * main_pred)
                                else:
                                    # Use main model
                                    val_preds_proba[i] = main_model.predict_proba(features.values.reshape(1, -1))[0][1]
                        else:
                            # Use main model for all predictions
                            val_preds_proba = main_model.predict_proba(X_val_reduced)[:, 1]
                    else:
                        # Use standard model (backward compatibility)
                        val_preds_proba = model.predict_proba(X_val_reduced)[:, 1]

                    if gender == "women's":
                        # Apply calibration
                        val_preds_calibrated = calibrate_by_expected_round(
                            val_preds_proba,
                            validation_df,
                            seed_diff_col='SeedDiff',
                            gender= "women's"
                        )
                    else:
                        # Apply calibration
                        val_preds_calibrated = calibrate_mens_predictions(
                            val_preds_proba,
                            validation_df,
                            seed_diff_col='SeedDiff'
                        )

                    # Calculate metrics
                    val_log_loss = log_loss(y_val, val_preds_proba)
                    val_calibrated_log_loss = log_loss(y_val, val_preds_calibrated)
                    val_auc = roc_auc_score(y_val, val_preds_proba)
                    val_brier = brier_score_loss(y_val, val_preds_proba)
                    val_calibrated_brier = brier_score_loss(y_val, val_preds_calibrated)

                    print(f"Validation Log Loss: {val_log_loss:.4f}")
                    print(f"Validation Calibrated Log Loss: {val_calibrated_log_loss:.4f}")
                    print(f"Validation AUC: {val_auc:.4f}")
                    print(f"Validation Brier Score: {val_brier:.4f}")
                    print(f"Validation Calibrated Brier Score: {val_calibrated_brier:.4f}")
                else:
                    print("No matching games found for validation season")

        # Return model and components for training mode
        return model, feature_cols, scaler, dropped_features

    # Generate predictions for prediction seasons
    elif is_prediction_mode:
        print(f"\n16. Making predictions for {gender}'s tournaments...")
        all_predictions = []

        # Check which prediction seasons have data
        available_prediction_seasons = [s for s in prediction_seasons if s in season_matchups]
        missing_seasons = [s for s in prediction_seasons if s not in season_matchups]

        if missing_seasons:
            print(f"WARNING: Missing data for seasons: {missing_seasons}")

        if not available_prediction_seasons:
            print("ERROR: No prediction seasons available in data")
            return pd.DataFrame(columns=['Season', 'Team1ID', 'Team2ID', 'Pred', 'MatchupID'])

        print(f"Proceeding with available seasons: {available_prediction_seasons}")

        # Check if we have a trained model
        if model is None:
            print("WARNING: No pre-trained model provided. Using seed-based predictions.")
            use_seed_based_model = True
        else:
            use_seed_based_model = False
            print(f"Using provided trained model: {type(model).__name__}")

            # If feature columns weren't provided but model exists, create a default set
            if feature_cols is None and hasattr(model, 'feature_names_in_'):
                feature_cols = model.feature_names_in_
                print(f"Using model's feature_names_in_: {len(feature_cols)} features")
            elif feature_cols is None:
                print("WARNING: No feature columns provided. Using basic feature set.")
                feature_cols = ['SeedDiff', 'WinRateDiff', 'OffEfficiencyDiff', 'DefEfficiencyDiff']

        # Process each season
        for season in available_prediction_seasons:
            print(f"Processing {season} tournament predictions...")

            # Get predictions for this season
            season_predictions = season_matchups[season].copy()
            print(f"Found {len(season_predictions)} matchups for season {season}")

            if len(season_predictions) == 0:
                print(f"WARNING: Season {season} has 0 matchups")
                continue

            if use_seed_based_model:
                # Use seed-based predictions
                if 'Team1Seed' in season_predictions.columns and 'Team2Seed' in season_predictions.columns:
                    print("Using seed-based prediction approach")
                    season_predictions['Pred'] = season_predictions.apply(
                        lambda row: calculate_seed_based_probability(row['Team1Seed'], row['Team2Seed']),
                        axis=1
                    )
                else:
                    print(f"ERROR: Cannot create seed-based predictions - missing seed columns")
                    season_predictions['Pred'] = 0.5  # Default to 50% if we can't do better
            else:
                # Use trained model
                # Check that all required feature columns exist
                missing_features = [col for col in feature_cols if col not in season_predictions.columns]
                if missing_features:
                    if len(missing_features) / len(feature_cols) > 0.5:  # If more than half features missing
                        print(f"ERROR: Too many missing features ({len(missing_features)}). Falling back to seed-based prediction.")
                        season_predictions['Pred'] = season_predictions.apply(
                            lambda row: calculate_seed_based_probability(row['Team1Seed'], row['Team2Seed']),
                            axis=1
                        )
                        all_predictions.append(season_predictions)
                        continue
                    else:
                        print(f"WARNING: Missing {len(missing_features)} feature columns. Proceeding with available features.")
                        available_features = [col for col in feature_cols if col in season_predictions.columns]
                        feature_cols = available_features

                try:
                    # Prepare prediction features 
                    prediction_features = season_predictions[feature_cols].copy()
                    
                    # Apply feature engineering
                    if scaler is not None:
                        prediction_features_scaled = scaler.transform(prediction_features)
                        prediction_features = pd.DataFrame(prediction_features_scaled, columns=feature_cols)
                    
                    prediction_features_enhanced = create_feature_interactions(prediction_features)
                    
                    # Remove dropped features if provided
                    if dropped_features:
                        prediction_features_reduced = prediction_features_enhanced.drop(
                            [f for f in dropped_features if f in prediction_features_enhanced.columns],
                            axis=1,
                            errors='ignore'
                        )
                    else:
                        prediction_features_reduced = prediction_features_enhanced
                    
                    # Check if we have a model dictionary with round-specific models
                    if isinstance(model, dict) and 'main_model' in model and 'round_models' in model:
                        main_model = model['main_model']
                        round_models = model['round_models']
                        
                        # Initialize predictions array
                        pred_proba = np.zeros(len(prediction_features_reduced))
                        
                        # Check if we have round information
                        if 'ExpectedRound' in season_predictions.columns:
                            print(f"Using round-specific models where available")
                            
                            # Make predictions for each sample
                            for i, (idx, features) in enumerate(prediction_features_reduced.iterrows()):
                                # Get expected round for this matchup
                                round_name = season_predictions.iloc[i]['ExpectedRound']
                                
                                if round_name in round_models:
                                    # Use round-specific model with weight
                                    round_pred = round_models[round_name].predict_proba(features.values.reshape(1, -1))[0][1]
                                    main_pred = main_model.predict_proba(features.values.reshape(1, -1))[0][1]
                                    
                                    # Blend with more weight to round-specific model in later rounds
                                    if round_name in ['Championship', 'Final4', 'Elite8', 'Sweet16']:
                                        blend_weight = 0.7  # 70% round-specific, 30% main model
                                    else:
                                        blend_weight = 0.5  # 50-50 blend for earlier rounds
                                        
                                    pred_proba[i] = (blend_weight * round_pred) + ((1-blend_weight) * main_pred)
                                else:
                                    # Use main model
                                    pred_proba[i] = main_model.predict_proba(features.values.reshape(1, -1))[0][1]
                        else:
                            # Use main model for all predictions
                            pred_proba = main_model.predict_proba(prediction_features_reduced)[:, 1]
                    else:
                        # Use standard model (backward compatibility)
                        pred_proba = model.predict_proba(prediction_features_reduced)[:, 1]

                    # Apply calibration
                    try:
                        pred_calibrated = calibrate_by_expected_round(
                            pred_proba,
                            season_predictions,
                            seed_diff_col='SeedDiff',
                            gender=gender
                        )
                    except Exception as e:
                        print(f"Error in calibration: {e}")
                        pred_calibrated = pred_proba

                    # Add predictions
                    season_predictions['Pred'] = pred_calibrated

                except Exception as e:
                    print(f"Error in prediction pipeline: {str(e)}")
                    print("Falling back to seed-based predictions")
                    season_predictions['Pred'] = season_predictions.apply(
                        lambda row: calculate_seed_based_probability(row['Team1Seed'], row['Team2Seed']),
                        axis=1
                    )

            # Run tournament simulations if we have the necessary data
            try:
                season_seed_data = df_seed[df_seed['Season'] == season]
                season_slots = df_tourney_slots[df_tourney_slots['Season'] == season]

                if len(season_seed_data) > 0 and len(season_slots) > 0:
                    print(f"Running tournament simulations for season {season}")
                    adjusted_predictions, _ = run_tournament_simulation_pre_tournament(
                        season_seed_data,
                        season_predictions,
                        season_slots,
                        num_simulations=1000
                    )

                    print(f"Adding {len(adjusted_predictions)} tournament-adjusted predictions for season {season}")
                    all_predictions.append(adjusted_predictions)
                else:
                    print(f"Cannot run tournament simulation, using direct predictions")
                    all_predictions.append(season_predictions)

            except Exception as e:
                print(f"Error in tournament simulation: {e}")
                print(f"Using direct predictions for season {season}")
                all_predictions.append(season_predictions)

        # Combine predictions if any were generated
        if all_predictions:
            try:
                print(f"Combining {len(all_predictions)} prediction sets")
                for i, pred_set in enumerate(all_predictions):
                    print(f"Prediction set {i+1}: {pred_set.shape}")
                    if 'Pred' not in pred_set.columns:
                        print(f"WARNING: 'Pred' column missing from prediction set {i+1}")

                final_predictions = pd.concat(all_predictions, ignore_index=True)
                print(f"Final prediction set contains {len(final_predictions)} predictions")
                return final_predictions
            except Exception as e:
                print(f"Error combining predictions: {str(e)}")
                # Create an emergency fallback prediction set
                if len(all_predictions) > 0:
                    print("Returning first prediction set as fallback")
                    return all_predictions[0]
                else:
                    # Return empty DataFrame with correct columns
                    return pd.DataFrame(columns=['Season', 'Team1ID', 'Team2ID', 'Pred', 'MatchupID'])
        else:
            print("WARNING: No predictions were generated for any season")
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=['Season', 'Team1ID', 'Team2ID', 'Pred', 'MatchupID'])
    else:
        print("ERROR: Invalid mode - must specify either training_seasons or prediction_seasons")
        return pd.DataFrame(columns=['Season', 'Team1ID', 'Team2ID', 'Pred', 'MatchupID'])