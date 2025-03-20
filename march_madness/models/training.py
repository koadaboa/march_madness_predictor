# Model training functions
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from march_madness.utils.data_access import get_data_with_index
from march_madness.data.loaders import filter_data_dict_by_seasons
from march_madness.models.evaluation import validate_model

def prepare_tournament_data_for_training(tourney_results, features_df, test_seasons=None):
    """
    Prepare tournament data for model training and validation.
    Ensures proper handling of train/test split to prevent data leakage.

    Args:
        tourney_results: DataFrame with tournament results
        features_df: DataFrame with matchup features
        test_seasons: List of seasons to use for testing

    Returns:
        Training and testing data splits
    """
    print("\nPreparing tournament data...")
    features_df_copy = features_df.copy()
    train_data = []
    found_matches = 0

    # Define features to use (exclude identifiers and target)
    feature_cols = [col for col in features_df_copy.columns if col not in
                   ['Season', 'Team1ID', 'Team2ID', 'Target', 'ExpectedRound', 'MatchupID']]

    if test_seasons:
        if not isinstance(test_seasons, list):
            test_seasons = [test_seasons]

        train_seasons = [season for season in features_df_copy['Season'].unique() if season not in test_seasons]
        print(f"Training seasons: {train_seasons}")
        print(f"Testing seasons: {test_seasons}")

        # For training data, match with actual tournament results
        for season in train_seasons:
            season_features = get_data_with_index(features_df_copy, 'Season', season, indexed_suffix='_by_season').copy()
            season_results = get_data_with_index(tourney_results, 'Season', season, indexed_suffix='_by_season')
            season_matches = 0

            for _, result in season_results.iterrows():
                # Find the matchup in our features dataset where Team1 won
                matchup1 = season_features[
                    (season_features['Team1ID'] == result['WTeamID']) &
                    (season_features['Team2ID'] == result['LTeamID'])
                ].copy()

                # Find the matchup in our features dataset where Team1 lost
                matchup2 = season_features[
                    (season_features['Team1ID'] == result['LTeamID']) &
                    (season_features['Team2ID'] == result['WTeamID'])
                ].copy()

                if len(matchup1) > 0:
                    matchup1['Target'] = 1  # Team1 won
                    train_data.append(matchup1)
                    season_matches += 1
                    found_matches += 1

                if len(matchup2) > 0:
                    matchup2['Target'] = 0  # Team1 lost
                    train_data.append(matchup2)
                    season_matches += 1
                    found_matches += 1

            print(f"Season {season}: Found {season_matches} matching feature records")

        print(f"Total matches found: {found_matches}")
        train_df = pd.concat(train_data, ignore_index=True)
        print(f"Training data shape: {train_df.shape}")

        # For validation, extract only the actual games that happened
        test_with_results = []
        test_results = tourney_results[tourney_results['Season'].isin(test_seasons)]

        for _, result in test_results.iterrows():
            # Find the matchup in our test dataset
            result_matchup_id = f"{result['Season']}_{min(result['WTeamID'], result['LTeamID'])}_{max(result['WTeamID'], result['LTeamID'])}"

            # Get all potential matchups that match this actual game
            test_features = features_df_copy[features_df_copy['Season'].isin(test_seasons)]
            test_features = test_features[test_features['MatchupID'] == result_matchup_id].copy()

            # Set target based on who actually won
            for _, row in test_features.iterrows():
                if row['Team1ID'] == result['WTeamID'] and row['Team2ID'] == result['LTeamID']:
                    row = row.copy()
                    row['Target'] = 1  # Team1 won
                    test_with_results.append(pd.DataFrame([row]))
                elif row['Team1ID'] == result['LTeamID'] and row['Team2ID'] == result['WTeamID']:
                    row = row.copy()
                    row['Target'] = 0  # Team1 lost
                    test_with_results.append(pd.DataFrame([row]))

        if test_with_results:
            test_with_results_df = pd.concat(test_with_results, ignore_index=True)
            print(f"Found {len(test_with_results_df)} actual matchups in test seasons for validation")

            X_train = train_df[feature_cols]
            y_train = train_df['Target']
            X_test = test_with_results_df[feature_cols]
            y_test = test_with_results_df['Target']
        else:
            print("No actual matchups found for test seasons")
            X_train = train_df[feature_cols]
            y_train = train_df['Target']
            X_test = pd.DataFrame(columns=feature_cols)
            y_test = pd.Series([])
    else:
        # Use random split if no test_seasons specified
        X = features_df_copy[feature_cols]
        y = features_df_copy['Target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    if len(X_test) > 0:
        X_test_scaled = scaler.transform(X_test)
    else:
        X_test_scaled = np.empty((0, len(feature_cols)))

    return X_train_scaled, X_test_scaled, y_train, y_test, X_train, X_test, scaler, feature_cols

def handle_class_imbalance(X_train, y_train, method='combined', random_state=42):
    """
    Handle class imbalance in the training data

    Args:
        X_train: Training feature data
        y_train: Training target data
        method: Method for handling imbalance
                ('smote', 'under', 'combined')
        random_state: Random state for reproducibility

    Returns:
        Tuple of (resampled X_train, resampled y_train)
    """
    # Check class distribution
    class_counts = pd.Series(y_train).value_counts()

    # If classes are already relatively balanced, return original data
    if min(class_counts) / max(class_counts) >= 0.4:
        return X_train, y_train

    # Apply the specified resampling method
    if method == 'smote':
        # SMOTE: Synthetic Minority Over-sampling Technique
        resampler = SMOTE(random_state=random_state)
        X_resampled, y_resampled = resampler.fit_resample(X_train, y_train)

    elif method == 'under':
        # Random Under-Sampling of the majority class
        resampler = RandomUnderSampler(random_state=random_state)
        X_resampled, y_resampled = resampler.fit_resample(X_train, y_train)

    elif method == 'combined':
        # Combined approach: SMOTE followed by under-sampling
        smote = SMOTE(sampling_strategy=0.8, random_state=random_state)
        under = RandomUnderSampler(sampling_strategy=0.9, random_state=random_state)

        # First apply SMOTE to oversample the minority class
        X_temp, y_temp = smote.fit_resample(X_train, y_train)

        # Then apply under-sampling to the majority class
        X_resampled, y_resampled = under.fit_resample(X_temp, y_temp)

    else:
        raise ValueError(f"Unknown resampling method: {method}")

    return X_resampled, y_resampled

# Modify train_ensemble_model in models/training.py
def train_ensemble_model(X_train, y_train, random_state=42):
    # XGBoost model - tune for higher precision
    xgb_model = XGBClassifier(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.7,
        gamma=0.2,
        reg_alpha=0.2,
        reg_lambda=1.5,
        min_child_weight=3,
        scale_pos_weight=1.0,
        random_state=random_state
    )

    # LightGBM model - tune for higher recall
    lgb_model = LGBMClassifier(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=5,
        num_leaves=24,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.2,
        reg_lambda=1.5,
        min_child_samples=20,
        class_weight='balanced',
        random_state=random_state
    )

    # Random Forest model
    rf_model = RandomForestClassifier(
        n_estimators=500,
        max_depth=12,
        min_samples_split=6,
        min_samples_leaf=3,
        max_features='sqrt',
        class_weight='balanced',
        random_state=random_state,
        n_jobs=-1
    )
    
    # Add a Logistic Regression model for better calibration
    logistic_model = LogisticRegression(
        C=0.8,
        class_weight='balanced',
        solver='liblinear',
        random_state=random_state
    )

    # Create a more diverse ensemble with optimized weights
    ensemble = VotingClassifier(
        estimators=[
            ('xgb', xgb_model),
            ('lgb', lgb_model),
            ('rf', rf_model),
            ('lr', logistic_model)
        ],
        voting='soft',
        weights=[0.4, 0.3, 0.2, 0.1]  # Give more weight to XGBoost and LightGBM
    )

    # Fit the ensemble
    ensemble.fit(X_train, y_train)

    return ensemble

def gender_specific_feature_selection(X, y, gender, importance_threshold=0.01):
    """
    Select features using a gender-specific approach combining importance scores
    with domain knowledge about basketball differences.
    
    Args:
        X: Feature DataFrame
        y: Target values
        gender: 'men' or 'women'
        importance_threshold: Minimum importance threshold (different by gender)
    
    Returns:
        Selected feature names
    """
    # Initial feature selection using XGBoost importance
    model = XGBClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Get feature importances
    importances = model.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Adjust threshold by gender
    if gender == "women's":
        # Women's tournaments show more predictable patterns, can be more selective
        adjusted_threshold = importance_threshold * 1.2
    else:
        # Men's tournaments need more features to capture variability
        adjusted_threshold = importance_threshold * 0.8
    
    # Filter by importance
    selected_features = feature_importance[feature_importance['importance'] > adjusted_threshold]['feature'].tolist()
    
    # Add gender-specific "must-include" features
    if gender == "women's":
        must_include = [
            # Women's game features with known importance
            'Team1FGPct', 'Team2FGPct', 'FGPctDiff',  # Overall shooting efficiency critical
            'OffEfficiencyDiff',       # Offensive efficiency differential 
            'Team1ASTtoTOV', 'Team2ASTtoTOV', # Ball control metrics
            'Womens_ScoringDist_2pt', 'Womens_ScoringDist_3pt', # Scoring distribution
            'Womens_AstToFGM',       # Team ball movement
        ]
    else:
        must_include = [
            # Men's game features with known importance
            'Team1DefEfficiency', 'Team2DefEfficiency', 'DefEfficiencyDiff', # Defense critical
            'Team1FG3Pct', 'Team2FG3Pct', # 3-point shooting crucial
            'Team1PressureScore', 'Team2PressureScore', # Performance under pressure
            'TourneyAppearancesDiff',  # Tournament experience
            'Team1RoundWinRate', 'Team2RoundWinRate', # Round-specific performance
            'CompositeUpsetScore',  # New feature for detecting upset potential
            'UpsetZone_MajorUpsetsV1',  # Zones of high upset probability
            'ThreePointUpsetFactor'  # 3-point shooting upset indicator
        ]
    
    # Ensure must-include features are in the selection
    for feature in must_include:
        if feature in X.columns and feature not in selected_features:
            selected_features.append(feature)
    
    # Always include these fundamental features regardless of gender
    core_features = ['SeedDiff', 'Team1Seed', 'Team2Seed', 'WinRateDiff', 'NetEfficiencyDiff']
    for feature in core_features:
        if feature in X.columns and feature not in selected_features:
            selected_features.append(feature)
    
    return selected_features

def time_based_cross_validation(data_dict, gender, feature_engineering_func, model_training_func):
    """
    Perform time-based cross-validation for tournament predictions
    
    Args:
        data_dict: Dictionary with all loaded data
        gender: 'men' or 'women'
        feature_engineering_func: Function to engineer features
        model_training_func: Function to train the model
        
    Returns:
        Cross-validation results and best model
    """
    # Define all seasons from 2010-2019 (excluding prediction seasons)
    all_seasons = list(range(2010, 2020))
    
    # We'll use rolling window validation with expanding training set
    cv_results = []
    
    # Start with minimum training size of 3 seasons
    min_train_seasons = 3
    
    for i in range(min_train_seasons, len(all_seasons)):
        # Training seasons are all seasons up to validation season
        train_seasons = all_seasons[:i]
        # Validation season is the next season
        val_season = all_seasons[i]
        
        print(f"CV fold {i-min_train_seasons+1}: Training on {train_seasons}, validating on {val_season}")
        
        # Prepare training data
        train_data = filter_data_dict_by_seasons(data_dict, train_seasons)
        train_features = feature_engineering_func(train_data, gender, train_seasons)
        
        # Prepare validation data
        val_data = filter_data_dict_by_seasons(data_dict, [val_season])
        val_features = feature_engineering_func(val_data, gender, [val_season])
        
        # Train model using the gender-specific training function
        model, feature_cols, scaler, dropped_features = model_training_func(
            train_features, gender, train_seasons, None, []
        )
        
        # Validate model
        val_metrics = validate_model(
            model, val_features, feature_cols, scaler, dropped_features, gender, val_season
        )
        
        if val_metrics:
            cv_results.append({
                'train_seasons': train_seasons,
                'val_season': val_season,
                'brier_score': val_metrics['brier_score'],
                'log_loss': val_metrics['log_loss'],
                'auc': val_metrics['auc']
            })
    
    # Find best model configuration based on Brier score
    if cv_results:
        cv_df = pd.DataFrame(cv_results)
        best_idx = cv_df['brier_score'].argmin()
        best_config = cv_results[best_idx]
        
        print(f"Best CV configuration: train on {best_config['train_seasons']}, " 
              f"validate on {best_config['val_season']}")
        print(f"Brier Score: {best_config['brier_score']:.4f}, "
              f"Log Loss: {best_config['log_loss']:.4f}, "
              f"AUC: {best_config['auc']:.4f}")
        
        # Train final model on all data
        all_data = filter_data_dict_by_seasons(data_dict, all_seasons)
        all_features = feature_engineering_func(all_data, gender, all_seasons)
        
        final_model, feature_cols, scaler, dropped_features = model_training_func(
            all_features, gender, all_seasons, None, []
        )
        
        return {
            'cv_results': cv_df,
            'model': final_model,
            'feature_cols': feature_cols,
            'scaler': scaler,
            'dropped_features': dropped_features
        }
    else:
        print("Cross-validation failed - no results")
        return None

def time_based_cross_validation(data_dict, gender, feature_engineering_func, model_training_func):
    """
    Perform time-based cross-validation for tournament predictions
    
    Args:
        data_dict: Dictionary with all loaded data
        gender: 'men' or 'women'
        feature_engineering_func: Function to engineer features
        model_training_func: Function to train the model
        
    Returns:
        Cross-validation results and best model
    """
    # Define all seasons from 2010-2019 (excluding prediction seasons)
    all_seasons = list(range(2010, 2020))
    
    # We'll use rolling window validation with expanding training set
    cv_results = []
    
    # Start with minimum training size of 3 seasons
    min_train_seasons = 3
    
    # Get tournament data from modeling_data
    tourney_data = data_dict.get('tourney_data', pd.DataFrame())
    if tourney_data.empty:
        print("ERROR: No tournament data available for cross-validation")
        return None
    
    # Get all available seasons from matchup data
    available_seasons = []
    season_matchups = data_dict.get('season_matchups', {})
    for season in season_matchups:
        try:
            available_seasons.append(int(season))
        except (ValueError, TypeError):
            continue
    
    available_seasons = sorted(available_seasons)
    print(f"Available seasons for CV: {available_seasons}")
    
    for i in range(min_train_seasons, len(available_seasons)):
        # Training seasons are all seasons up to validation season
        train_seasons = available_seasons[:i]
        # Validation season is the next season
        val_season = available_seasons[i]
        
        print(f"CV fold {i-min_train_seasons+1}: Training on {train_seasons}, validating on {val_season}")
        
        # Filter tournament results for training
        train_tourney = tourney_data[tourney_data['Season'].isin(train_seasons)]
        
        # Create training features using modeling_data
        train_features = {}
        for season in train_seasons:
            if season in season_matchups:
                train_features[season] = season_matchups[season]
        
        # Validate model on the next season
        val_results = None
        if val_season in season_matchups:
            # Get validation features
            val_features = season_matchups[val_season]
            
            # Get actual tournament results for this season
            val_tourney = tourney_data[tourney_data['Season'] == val_season]
            
            if not val_tourney.empty:
                # Now we can evaluate our model on this season
                # We'll need to implement a simplified version of train_and_predict_model here
                # that doesn't require the original raw data_dict
                pass
    
    # Find best model configuration based on Brier score
    if cv_results:
        cv_df = pd.DataFrame(cv_results)
        best_idx = cv_df['brier_score'].argmin()
        best_config = cv_results[best_idx]
        
        print(f"Best CV configuration: train on {best_config['train_seasons']}, " 
              f"validate on {best_config['val_season']}")
        print(f"Brier Score: {best_config['brier_score']:.4f}, "
              f"Log Loss: {best_config['log_loss']:.4f}, "
              f"AUC: {best_config['auc']:.4f}")
        
        # Train final model on all data
        all_data = filter_data_dict_by_seasons(data_dict, all_seasons)
        all_features = feature_engineering_func(all_data, gender, all_seasons)
        
        final_model, feature_cols, scaler, dropped_features = model_training_func(
            all_features, gender, all_seasons, None, []
        )
        
        return {
            'cv_results': cv_df,
            'model': final_model,
            'feature_cols': feature_cols,
            'scaler': scaler,
            'dropped_features': dropped_features
        }
    else:
        print("Cross-validation failed - no results")
        return None

def create_mens_specific_model(random_state=3):
    """
    Create a men's tournament-specific model with enhanced upset detection
    """
    # Highly tuned SVM for upset detection
    upset_svm = SVC(
        C=1.5,  # More flexibility
        kernel='rbf',
        probability=True,
        class_weight={0: 1, 1: 4},  # Much higher weight for minority class (upsets)
        gamma='scale',  # Helps with non-linear decision boundaries
        random_state=random_state
    )
    
    # XGBoost with aggressive upset learning
    xgb_model = XGBClassifier(
        n_estimators=1000,  # More trees
        learning_rate=0.02,  # Slower learning
        max_depth=5,  # Slightly deeper trees
        subsample=0.8,
        colsample_bytree=0.7,
        gamma=0.3,  # More conservative tree growth
        reg_alpha=0.3,  # L1 regularization
        reg_lambda=1.7,  # L2 regularization
        min_child_weight=4,  # More conservative splits
        scale_pos_weight=3.5,  # Strong focus on minority class (upsets)
        random_state=random_state
    )
    
    # Gradient Boosting with upset sensitivity
    gb_model = GradientBoostingClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        min_samples_leaf=3,
        min_samples_split=7,
        subsample=0.85,
        max_features='sqrt',
        random_state=random_state
    )
    
    # Random Forest with upset-aware configuration
    rf_model = RandomForestClassifier(
        n_estimators=600,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced_subsample',
        random_state=random_state,
        n_jobs=-1
    )
    
    # Logistic Regression with careful regularization
    logistic_model = LogisticRegression(
        C=0.7,  # Less regularization
        class_weight='balanced',
        solver='liblinear',
        penalty='l1',  # Sparse feature selection
        random_state=random_state
    )
    
    # Ensemble with carefully tuned weights
    return VotingClassifier(
        estimators=[
            ('xgb', xgb_model),     # Strong predictive power
            ('svm', upset_svm),     # Sensitive to upsets
            ('rf', rf_model),       # Robust to variations
            ('gb', gb_model),       # Captures complex interactions
            ('lr', logistic_model)  # Linear baseline
        ],
        voting='soft',
        weights=[0.35, 0.25, 0.20, 0.15, 0.05]  # Prioritize upset-sensitive models
    )

def create_womens_specific_model(random_state=3):
    # XGBoost configured for women's tournament patterns
        xgb_model = XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=5,    # Slightly deeper trees to capture stronger patterns
            subsample=0.85,
            colsample_bytree=0.75,
            gamma=0.15,
            reg_alpha=0.25,
            reg_lambda=1.0,
            min_child_weight=2,
            scale_pos_weight=1.0,
            random_state=random_state
        )
        
        # LightGBM with women's-specific parameters
        lgb_model = LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,     # Deeper trees to capture stronger patterns
            num_leaves=32,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=0.15,
            reg_lambda=1.2,
            min_child_samples=15,
            class_weight='balanced',
            random_state=random_state
        )
        
        # Random Forest - more trees, deeper
        rf_model = RandomForestClassifier(
            n_estimators=400,
            max_depth=10,
            min_samples_split=8,
            min_samples_leaf=4,
            max_features='sqrt',
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1
        )
        
        # Linear model for capturing direct relationships
        logistic_model = LogisticRegression(
            C=0.9,
            class_weight='balanced',
            solver='liblinear',
            random_state=random_state
        )
        
        # Create ensemble with women's-specific weights
        # Give more weight to XGBoost which handles strong feature signals well
        return VotingClassifier(
            estimators=[
                ('xgb', xgb_model),
                ('lgb', lgb_model),
                ('rf', rf_model),
                ('lr', logistic_model)
            ],
            voting='soft',
            weights=[0.45, 0.25, 0.15, 0.15]
        )

def train_round_specific_models(X_train, y_train, tournament_rounds, gender):
    """
    Train specialized models for each tournament round with advanced optimization techniques
    
    Key Improvements:
    1. More sophisticated model selection for critical rounds
    2. Enhanced sampling techniques
    3. Dynamic hyperparameter tuning based on round characteristics
    4. Detailed logging of model performance
    """
    round_models = {}
    
    # Define critical rounds with round-specific challenges
    critical_rounds = {
        "men's": {
            'Round32': {'accuracy_baseline': 0.625, 'complexity': 'high_variance'},
            'Sweet16': {'accuracy_baseline': 0.375, 'complexity': 'upset_prone'},
            'Elite8': {'accuracy_baseline': 0.750, 'complexity': 'moderate'},
            'Final4': {'accuracy_baseline': 0.500, 'complexity': 'high_stakes'},
            'Championship': {'accuracy_baseline': 0.500, 'complexity': 'critical'}
        },
        "women's": {
            'Final4': {'accuracy_baseline': 0.333, 'complexity': 'high_variance'},
            'Championship': {'accuracy_baseline': 0.500, 'complexity': 'critical'}
        }
    }
    
    # Get round-specific configurations
    round_configs = critical_rounds.get(gender, {})
    
    for round_name in tournament_rounds:
        print(f"Training specialized model for {round_name}...")
        
        # Filter training data for this round
        round_mask = X_train['ExpectedRound'] == round_name
        X_round = X_train[round_mask]
        y_round = y_train[round_mask]
        
        # Skip rounds with insufficient data
        if len(X_round) < 15:
            print(f"Insufficient data for {round_name}, using general model")
            continue
        
        # Advanced sampling for class balance
        from imblearn.over_sampling import SMOTE
        from imblearn.under_sampling import RandomUnderSampler
        from sklearn.model_selection import train_test_split
        
        # Combine SMOTE and Undersampling for better class balance
        if round_name in round_configs:
            # More aggressive sampling for challenging rounds
            if len(y_round) > 10 and sum(y_round) > 3 and len(y_round) - sum(y_round) > 3:
                # Stratified split to maintain round-specific data distribution
                X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                    X_round, y_round, stratify=y_round, test_size=0.2, random_state=42
                )
                
                # Combined sampling strategy
                smote = SMOTE(sampling_strategy=0.8, random_state=42)
                under = RandomUnderSampler(sampling_strategy=0.9, random_state=42)
                
                X_resampled, y_resampled = smote.fit_resample(X_train_split, y_train_split)
                X_resampled, y_resampled = under.fit_resample(X_resampled, y_resampled)
                
                print(f"  Applied advanced sampling for {round_name} - now {len(X_resampled)} samples")
            else:
                X_resampled, y_resampled = X_round, y_round
        else:
            X_resampled, y_resampled = X_round, y_round
        
        # Dynamic model selection based on round complexity
        if round_name in round_configs:
            config = round_configs[round_name]
            
            if gender == "men's" and round_name == 'Sweet16':
                # Specialized XGBoost for upset-prone rounds
                model = XGBClassifier(
                    n_estimators=500,
                    learning_rate=0.02,
                    max_depth=5,
                    subsample=0.85,
                    colsample_bytree=0.8,
                    scale_pos_weight=2.5 if config['complexity'] == 'upset_prone' else 1.5,
                    gamma=0.3,  # More conservative tree growth
                    min_child_weight=3,
                    random_state=42
                )
            elif round_name in ['Final4', 'Championship']:
                # More conservative models for high-stakes rounds
                model = GradientBoostingClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=4,
                    subsample=0.9,  # Higher subsample for stability
                    random_state=42
                )
            else:
                # Default: Advanced Random Forest
                model = RandomForestClassifier(
                    n_estimators=400,
                    max_depth=6,
                    min_samples_split=5,
                    class_weight='balanced_subsample',
                    random_state=42
                )
        else:
            # Fallback to standard Random Forest
            model = RandomForestClassifier(
                n_estimators=300,
                max_depth=6,
                min_samples_split=5,
                class_weight='balanced',
                random_state=42
            )
        
        # Train the model
        model.fit(X_resampled, y_resampled)
        round_models[round_name] = model
        
        # Comprehensive model evaluation
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        train_preds = model.predict(X_resampled)
        
        # Extended performance metrics
        accuracy = accuracy_score(y_resampled, train_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(y_resampled, train_preds, average='binary')
        
        print(f"  {round_name} model performance:")
        print(f"    Training Accuracy: {accuracy:.4f}")
        print(f"    Precision: {precision:.4f}")
        print(f"    Recall: {recall:.4f}")
        print(f"    F1 Score: {f1:.4f}")
    
    return round_models

def accuracy_maximizing_push(predictions, threshold=0.5):
    """
    Make a more balanced push that preserves some upset detection ability
    """
    pushed_preds = np.copy(predictions)
    
    for i in range(len(pushed_preds)):
        original = pushed_preds[i]
        
        # If it's a strong favorite (high probability), push it higher but more gently
        if original > 0.65:  # Clear favorite
            pushed_preds[i] = min(0.95, original + 0.05)
        # If it's a moderate favorite, push slightly
        elif original > 0.55:
            pushed_preds[i] = min(0.90, original + 0.03)
        # If it's a strong underdog prediction, preserve it
        elif original < 0.33:
            pushed_preds[i] = max(0.05, original - 0.05)  # Boost underdog confidence
        # If it's close to 0.5 but slightly favoring underdog, preserve some upset potential
        elif 0.33 <= original < 0.47:
            # Don't modify these - these are our potential upset predictions
            pass
        # If it's very close to 0.5, make a small push based on direction
        elif 0.47 <= original < 0.53:
            # Make a small push based on which side of 0.5 we're on
            if original >= 0.5:
                pushed_preds[i] = original + 0.03
            else:
                pushed_preds[i] = original - 0.03
    
    return pushed_preds

def adjust_for_upset_potential(predictions, X_test):
    """Specifically designed to preserve some upset prediction capability"""
    adjusted_preds = np.copy(predictions)
    
    if 'Team1Seed' in X_test.columns and 'Team2Seed' in X_test.columns:
        for i in range(len(adjusted_preds)):
            team1_seed = X_test.iloc[i]['Team1Seed']
            team2_seed = X_test.iloc[i]['Team2Seed']
            
            # Classic upset matchups
            classic_upset = ((team1_seed == 12 and team2_seed == 5) or
                            (team1_seed == 11 and team2_seed == 6) or
                            (team1_seed == 10 and team2_seed == 7) or
                            (team1_seed == 9 and team2_seed == 8))
            
            # If this is a classic upset matchup and prediction is strongly against upset
            if classic_upset and adjusted_preds[i] > 0.7:
                # Reduce the confidence to give some chance for upset
                adjusted_preds[i] = 0.65
            
            # If prediction is somewhat close to upset threshold
            if classic_upset and 0.45 <= adjusted_preds[i] <= 0.6:
                # Pull closer to 0.5 to give some upset potential
                adjusted_preds[i] = 0.48
    
    return adjusted_preds