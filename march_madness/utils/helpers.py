import pandas as pd
import numpy as np
import random
import warnings
warnings.filterwarnings("ignore")

def prepare_modeling_data(data_dict, gender, starting_season, current_season, seasons_to_process):
    """
    Performs feature engineering and creates prediction datasets (Steps 1-9)

    Args:
        data_dict: Dictionary containing the loaded datasets
        gender: 'men' or 'women' to identify which dataset is being processed
        starting_season: First season to include in analysis
        current_season: Current season for prediction
        seasons_to_process: List of seasons to process

    Returns:
        Dictionary with all the prepared data needed for modeling
    """
    print(f"\n=== Preparing modeling data for {gender}'s NCAA Basketball Tournament ===\n")

    # Add debugging for input data
    if 'df_tourney' in data_dict:
        print(f"DEBUG: {gender} tourney data contains seasons:", sorted(data_dict['df_tourney']['Season'].unique()))
        for season in seasons_to_process:
            games = data_dict['df_tourney'][data_dict['df_tourney']['Season'] == season]
            if len(games) > 0:
                print(f"WARNING: Input data contains {len(games)} actual tournament results for season {season}")
    else:
        print(f"DEBUG: No tournament data in input data_dict")

    # Extract individual dataframes from the dictionary with error handling
    df_tourney = data_dict.get('df_tourney', pd.DataFrame())
    df_reg = data_dict.get('df_regular', pd.DataFrame())
    df_seed = data_dict.get('df_seed', pd.DataFrame())
    df_teams = data_dict.get('df_teams', pd.DataFrame())
    df_team_conferences = data_dict.get('df_team_conferences', pd.DataFrame())
    df_tourney_slots = data_dict.get('df_tourney_slots', pd.DataFrame())
    df_conf_tourney = data_dict.get('df_conf_tourney', pd.DataFrame())
    df_coaches = data_dict.get('df_coaches', pd.DataFrame())
    df_seed_round_slots = data_dict.get('df_seed_round_slots', pd.DataFrame())

    # Create MatchupID for tournament data if not already present
    if 'MatchupID' not in df_tourney.columns and not df_tourney.empty:
        df_tourney['MatchupID'] = df_tourney.apply(
            lambda row: f"{row['Season']}_{min(row['WTeamID'], row['LTeamID'])}_{max(row['WTeamID'], row['LTeamID'])}",
            axis=1
        )

    # Determine tournament days by season to avoid data leakage
    tournament_days = {}
    for season in df_tourney['Season'].unique():
        tournament_days[season] = df_tourney[df_tourney['Season'] == season]['DayNum'].unique()

    # SAFEGUARD: Apply data leakage prevention before using tournament data
    def apply_data_leakage_safeguard(data_df, season_col='Season', current_season=None):
        """
        Apply consistent safeguard against data leakage from future seasons

        Args:
            data_df: DataFrame to filter
            season_col: Name of the season column
            current_season: Current season being processed

        Returns:
            Filtered DataFrame
        """
        if data_df.empty or current_season is None:
            return data_df

        is_prediction_season = current_season in seasons_to_process

        if is_prediction_season:
            # Only use tournament data from before the prediction seasons
            filtered_df = data_df[~data_df[season_col].isin(seasons_to_process)]

            # Only print warning if we actually filtered something
            if len(filtered_df) < len(data_df):
                print(f"SAFEGUARD: Removed {len(data_df) - len(filtered_df)} rows of future season data")

            return filtered_df
        else:
            return data_df

    # Apply safeguard to tournament data
    safe_tourney_df = apply_data_leakage_safeguard(df_tourney, 'Season', current_season)
    print(f"After safeguard: Tournament data contains seasons {sorted(safe_tourney_df['Season'].unique())}")

    # Filter regular season to exclude tournament days for the current season
    filtered_reg = filter_reg_season(df_reg, current_season, tournament_days)

    print("\n1. Creating team season profiles...")
    team_profiles, all_team_games = create_team_season_profiles(
        filtered_reg,
        current_season=current_season,
        tournament_days=tournament_days
    )
    print(f"Created profiles for {len(team_profiles)} team-seasons")

    print("\n2. Calculating momentum features...")
    momentum_data = calculate_momentum_features(
        all_team_games,
        current_season=current_season,
        tournament_days=tournament_days
    )
    print(f"Generated momentum features for {len(momentum_data)} team-seasons")

    # Coach features
    if not df_coaches.empty:
        print("\n3. Calculating coach features...")
        # SAFEGUARD: Use safe_tourney_df instead of df_tourney
        coach_features = calculate_coach_features(df_coaches, safe_tourney_df)
        print(f"Generated coach features for {len(coach_features)} team-seasons")
    else:
        coach_features = pd.DataFrame(columns=['Season', 'TeamID', 'CoachName', 'CoachYearsExp', 'CoachTourneyExp', 'CoachChampionships'])
        print("No coach data available, using empty coach features")

    print("\n4. Calculating tournament history...")
    # SAFEGUARD: Use safe_tourney_df instead of df_tourney
    tourney_history = calculate_tournament_history(safe_tourney_df, current_season=current_season)
    print(f"Calculated tournament history for {len(tourney_history)} team-seasons")

    print("\n5. Calculating conference strength...")
    # SAFEGUARD: Use safe_tourney_df instead of df_tourney
    conf_strength = calculate_conference_strength(df_team_conferences, safe_tourney_df, df_seed, current_season=current_season)
    print(f"Calculated conference strength for {len(conf_strength)} conference-seasons")

    print("\n6. Enhancing team metrics...")
    enhanced_team_games, enhanced_team_profiles = enhance_team_metrics(
        all_team_games,
        team_profiles,
        filtered_reg
    )
    print(f"Enhanced metrics for {len(enhanced_team_profiles)} team-seasons")

    print("\n7. Calculating strength of schedule...")
    sos_data = calculate_strength_of_schedule(
        filtered_reg,
        enhanced_team_profiles,
        current_season=current_season,
        tournament_days=tournament_days
    )
    print(f"Calculated SOS for {len(sos_data)} team-seasons")

    print("\n8. Calculating tournament-specific features...")
    # Calculate tournament features
    all_round_performance = []
    all_pressure_metrics = []
    all_conf_impact = []
    all_seed_features = []
    all_coach_metrics = []

    for season in sorted(set(safe_tourney_df['Season'].unique()) | set(seasons_to_process)):
        print(f"  Calculating features for season {season}...")

        # Filter data to avoid leakage
        historical_tourney = safe_tourney_df[safe_tourney_df['Season'] < season]
        is_first_season = (len(historical_tourney) == 0)

        if is_first_season:
            print(f"  Processing first available season ({season}) - using seed-based features")
            season_seed_data = df_seed[df_seed['Season'] == season]
            
            # Get all teams for this season from team profiles
            season_profiles = enhanced_team_profiles[enhanced_team_profiles['Season'] == season]
            all_teams = season_profiles['TeamID'].unique()
            
            # Call our modified functions with all required arguments
            all_round_performance.append(create_seed_based_features(all_teams, season_seed_data, season))
            all_pressure_metrics.append(create_seed_based_pressure_metrics(all_teams, season_seed_data, season))
            all_seed_features.append(create_seed_based_trend_features(all_teams, season_seed_data, season))
            continue


        # Calculate features using historical data
        all_round_performance.append(calculate_expected_round_features(df_seed, historical_tourney, current_season=season))

        if not df_conf_tourney.empty:
            season_conf_tourney = df_conf_tourney[df_conf_tourney['Season'] <= season]
            # SAFEGUARD: Apply safeguard to conference tourney data
            safe_conf_tourney = apply_data_leakage_safeguard(season_conf_tourney, 'Season', season)
            all_conf_impact.append(calculate_conference_tournament_impact(
                safe_conf_tourney, df_team_conferences, filtered_reg, current_season=season
            ))

        all_seed_features.append(create_seed_trend_features(df_seed, historical_tourney, current_season=season))

        # SAFEGUARD: Only use historical coach data for tournament metrics
        if not df_coaches.empty:
            safe_coaches = apply_data_leakage_safeguard(df_coaches, 'Season', season)
            all_coach_metrics.append(calculate_coach_tournament_metrics(historical_tourney, safe_coaches, current_season=season))

    # Combine features with safety checks
    round_performance = pd.concat(all_round_performance, ignore_index=True) if all_round_performance else pd.DataFrame()
    pressure_metrics = pd.concat(all_pressure_metrics, ignore_index=True) if all_pressure_metrics else pd.DataFrame()
    conf_impact = pd.concat(all_conf_impact, ignore_index=True) if all_conf_impact else None
    seed_features = pd.concat(all_seed_features, ignore_index=True) if all_seed_features else pd.DataFrame()
    coach_metrics = pd.concat(all_coach_metrics, ignore_index=True) if all_coach_metrics else None

    print(f"Calculated tournament-specific features for all seasons")

    print("\n9. Creating matchup features...")
    # Create separate matchup datasets for each season
    season_matchups = {}

    for season in seasons_to_process:
        print(f"Creating prediction dataset for season {season}")

        # Get ALL teams for this season from team profiles
        season_team_profiles = enhanced_team_profiles[enhanced_team_profiles['Season'] == season]
        all_teams = season_team_profiles['TeamID'].unique()
        
        # Get tournament teams
        season_seed_data = df_seed[df_seed['Season'] == season]
        
        # For seasons without existing tournament features, create them for all teams
        if len(round_performance[round_performance['Season'] == season]) == 0:
            season_round_perf = create_seed_based_features(all_teams, season_seed_data, season)
            round_performance = pd.concat([round_performance, season_round_perf], ignore_index=True)
            
        if len(pressure_metrics[pressure_metrics['Season'] == season]) == 0:
            season_pressure = create_seed_based_pressure_metrics(all_teams, season_seed_data, season)
            pressure_metrics = pd.concat([pressure_metrics, season_pressure], ignore_index=True)
            
        if len(seed_features[seed_features['Season'] == season]) == 0:
            season_seed_features = create_seed_based_trend_features(all_teams, season_seed_data, season)
            seed_features = pd.concat([seed_features, season_seed_features], ignore_index=True)

        # Create matchup features using ALL teams
        season_data = create_tournament_prediction_dataset(
            [season],
            enhanced_team_profiles,
            df_seed,
            momentum_data,
            sos_data,
            coach_features,
            tourney_history,
            conf_strength,
            df_team_conferences,
            enhanced_team_profiles,
            enhanced_team_profiles,
            round_performance,
            pressure_metrics,
            conf_impact,
            seed_features,
            coach_metrics
        )

        # Add matchup IDs
        season_data['MatchupID'] = season_data.apply(
            lambda row: f"{row['Season']}_{min(row['Team1ID'], row['Team2ID'])}_{max(row['Team1ID'], row['Team2ID'])}",
            axis=1
        )

        # Store season data
        season_matchups[season] = season_data
        print(f"Created {len(season_data)} matchup features for season {season}")

    # Prepare final return data structure
    return_data = {
        'season_matchups': season_matchups,
        'tourney_data': safe_tourney_df,
        'enhanced_team_profiles': enhanced_team_profiles,
        'round_performance': round_performance,
        'pressure_metrics': pressure_metrics,
        'conf_impact': conf_impact,
        'seed_features': seed_features,
        'coach_metrics': coach_metrics,
        'df_seed': df_seed,
        'df_teams': df_teams,
        'df_tourney_slots': df_tourney_slots
    }

    return return_data

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

        # Define features to use (exclude identifiers and target)
        feature_cols = [col for col in training_df.columns if col not in
                      ['Season', 'Team1ID', 'Team2ID', 'Target', 'ExpectedRound', 'MatchupID']]

        # Split features and target
        X_train = training_df[feature_cols]
        y_train = training_df['Target']

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Create feature interactions
        print("\n11. Creating feature interactions...")
        X_train_enhanced = create_feature_interactions(X_train)
        print(f"Enhanced features: {X_train_enhanced.shape}")

        # Drop highly correlated features
        print("\n12. Removing redundant features...")
        X_train_reduced, dropped_features = drop_redundant_features(X_train_enhanced, threshold=0.95)
        print(f"Reduced features: {X_train_reduced.shape}, dropped {len(dropped_features)} redundant features")

        # Handle class imbalance
        print("\n13. Handling class imbalance...")
        X_train_balanced, y_train_balanced = handle_class_imbalance(
            X_train_reduced, y_train, method='combined', random_state=42
        )
        print(f"Balanced training data: {X_train_balanced.shape}")

        # Train model
        print(f"\n14. Training ensemble model for {gender}'s tournaments...")
        model = train_ensemble_model(X_train_balanced, y_train_balanced, random_state=3)
        print("Model trained")

        # Validate model if validation season is provided
        if validation_season:
            print(f"\n15. Validating model on {validation_season} data...")

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
                    val_preds_proba = model.predict_proba(X_val_reduced)[:, 1]

                    # Apply calibration
                    val_preds_calibrated = calibrate_by_expected_round(
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

                    # Make predictions
                    try:
                        pred_proba = model.predict_proba(prediction_features_reduced)[:, 1]
                        print(f"Successfully generated model predictions for {len(prediction_features_reduced)} matchups")
                    except Exception as e:
                        print(f"Error in model prediction: {e}")
                        print("Falling back to seed-based predictions")
                        pred_proba = season_predictions.apply(
                            lambda row: calculate_seed_based_probability(row['Team1Seed'], row['Team2Seed']),
                            axis=1
                        ).values

                    # Apply calibration
                    try:
                        pred_calibrated = calibrate_by_expected_round(
                            pred_proba,
                            season_predictions,
                            seed_diff_col='SeedDiff'
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

