# march_madness/models/common.py

import pandas as pd
import numpy as np

def create_feature_interactions(X, important_features=None, max_degree=2):
    """
    Create meaningful feature interactions

    Args:
        X: DataFrame with features
        important_features: List of most important feature names
        max_degree: Maximum degree for interactions

    Returns:
        DataFrame with original features and interactions
    """
    # Start with the original data
    X_enhanced = X.copy()

    # If no important features provided, use all features
    if important_features is None:
        important_features = X.columns.tolist()

    # Select top features for interactions (to avoid explosion of features)
    if len(important_features) > 30:
        interaction_features = important_features[:30]
    else:
        interaction_features = important_features

    # Create interaction terms for pairs of important features
    for i, feat1 in enumerate(interaction_features):
        if feat1 not in X.columns:
            continue

        # 1. Create squared terms for certain features (only for numeric columns)
        if X[feat1].dtype.kind in 'ifc':  # integer, float, or complex
            feat_name = f"{feat1}_squared"
            X_enhanced[feat_name] = X[feat1] ** 2
        
        # 2. Create interaction terms with other important features (only for numeric columns)
        for j in range(i+1, len(interaction_features)):
            feat2 = interaction_features[j]

            if feat2 not in X.columns:
                continue
                
            # Only create interactions between numeric features
            if X[feat1].dtype.kind in 'ifc' and X[feat2].dtype.kind in 'ifc':
                feat_name = f"{feat1}_x_{feat2}"
                X_enhanced[feat_name] = X[feat1] * X[feat2]

    # Create specific basketball-relevant interactions - add type checking for each
    # Efficiency interactions
    if all(f in X.columns for f in ['Team1OffEfficiency', 'Team2DefEfficiency']):
        # Check if both columns are numeric
        if (X['Team1OffEfficiency'].dtype.kind in 'ifc' and 
            X['Team2DefEfficiency'].dtype.kind in 'ifc' and
            (X['Team2DefEfficiency'] != 0).all()):  # Avoid division by zero
            X_enhanced['Team1Off_vs_Team2Def'] = X['Team1OffEfficiency'] / X['Team2DefEfficiency']

    if all(f in X.columns for f in ['Team2OffEfficiency', 'Team1DefEfficiency']):
        # Check if both columns are numeric
        if (X['Team2OffEfficiency'].dtype.kind in 'ifc' and 
            X['Team1DefEfficiency'].dtype.kind in 'ifc' and
            (X['Team1DefEfficiency'] != 0).all()):  # Avoid division by zero
            X_enhanced['Team2Off_vs_Team1Def'] = X['Team2OffEfficiency'] / X['Team1DefEfficiency']

    # Seed interactions with other factors
    if all(f in X.columns for f in ['SeedDiff', 'WinRateDiff']):
        # Check if both columns are numeric
        if (X['SeedDiff'].dtype.kind in 'ifc' and X['WinRateDiff'].dtype.kind in 'ifc'):
            X_enhanced['SeedDiff_x_WinRateDiff'] = X['SeedDiff'] * X['WinRateDiff']

    if all(f in X.columns for f in ['SeedDiff', 'SOSPercentileDiff']):
        # Check if both columns are numeric
        if (X['SeedDiff'].dtype.kind in 'ifc' and X['SOSPercentileDiff'].dtype.kind in 'ifc'):
            X_enhanced['SeedDiff_x_SOSDiff'] = X['SeedDiff'] * X['SOSPercentileDiff']

    # Experience interactions
    if all(f in X.columns for f in ['TourneyAppearancesDiff', 'SeedDiff']):
        # Check if both columns are numeric
        if (X['TourneyAppearancesDiff'].dtype.kind in 'ifc' and X['SeedDiff'].dtype.kind in 'ifc'):
            X_enhanced['SeedDiff_x_ExperienceDiff'] = X['SeedDiff'] * X['TourneyAppearancesDiff']

    # Tournament readiness interaction with seed
    if all(f in X.columns for f in ['Team1TournamentReadiness', 'Team2TournamentReadiness', 'SeedDiff']):
        # Check if all columns are numeric
        if (X['Team1TournamentReadiness'].dtype.kind in 'ifc' and 
            X['Team2TournamentReadiness'].dtype.kind in 'ifc' and
            X['SeedDiff'].dtype.kind in 'ifc'):
            X_enhanced['TournamentReadiness_x_SeedDiff'] = (
                X['Team1TournamentReadiness'] - X['Team2TournamentReadiness']) * X['SeedDiff']
    
    # Handle potential division by zero issues
    for col in X_enhanced.columns:
        if col not in X.columns:  # Only process new columns
            X_enhanced[col] = X_enhanced[col].replace([np.inf, -np.inf], np.nan).fillna(0)

    return X_enhanced

def drop_redundant_features(X, threshold=0.95):
    """
    Drop highly correlated features to reduce multicollinearity

    Args:
        X: DataFrame with features
        threshold: Correlation threshold for dropping

    Returns:
        DataFrame with reduced features
    """
    # Make a copy of the input dataframe
    X_copy = X.copy()
    
    # Identify non-numeric columns
    numeric_cols = X_copy.select_dtypes(include=['int', 'float']).columns.tolist()
    non_numeric_cols = [col for col in X_copy.columns if col not in numeric_cols]
    
    print(f"Found {len(numeric_cols)} numeric columns and {len(non_numeric_cols)} non-numeric columns")
    
    # If we have no numeric columns, return the original dataframe
    if len(numeric_cols) == 0:
        return X_copy, []
    
    # Use only numeric columns for correlation calculation
    X_numeric = X_copy[numeric_cols]
    
    # Calculate correlation matrix
    corr_matrix = X_numeric.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find features with correlation greater than threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    print(f"Dropping {len(to_drop)} highly correlated features with threshold {threshold}")

    # Drop highly correlated features from the numeric columns
    X_numeric_reduced = X_numeric.drop(to_drop, axis=1)
    
    # Combine the reduced numeric features with the non-numeric columns
    if non_numeric_cols:
        X_reduced = pd.concat([X_numeric_reduced, X_copy[non_numeric_cols]], axis=1)
    else:
        X_reduced = X_numeric_reduced

    return X_reduced, to_drop