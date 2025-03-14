import pandas as pd
import numpy as np
from march_madness.utils.data_access import optimize_dataframes_with_indexes

def filter_reg_season(df, current_season, tournament_days):
    """
    Filter dataframe to exclude tournament days for the current season
    to prevent data leakage

    Args:
        df: DataFrame to filter
        current_season: Current season being processed
        tournament_days: Dictionary of tournament days by season

    Returns:
        Filtered DataFrame
    """
    if current_season and tournament_days is not None:
        # Get tournament days for the current season
        current_season_tourney_days = tournament_days.get(current_season, [])

        if len(current_season_tourney_days) > 0:
            df = df[
                (df['Season'] < current_season) |
                ((df['Season'] == current_season) &
                 (~df['DayNum'].isin(current_season_tourney_days)))
            ]

        return df
    else:
        return df

def load_and_filter_data():
    """
    Load and filter data to prevent data leakage
    """
    print("Loading men's and women's data...")

    # Load all data from 2015 onwards
    full_mens_data = load_mens_data(STARTING_SEASON)
    full_womens_data = load_womens_data(STARTING_SEASON)

     # Apply indexing optimization to the loaded data
    full_mens_data = optimize_dataframes_with_indexes(full_mens_data)
    full_womens_data = optimize_dataframes_with_indexes(full_womens_data)

    # Create separate dictionaries for training data (2015-2019) and prediction data (2021-2024)
    mens_train_data = filter_data_dict_by_seasons(full_mens_data,
                                                 seasons_to_include=TRAINING_SEASONS + [VALIDATION_SEASON],
                                                 seasons_to_exclude=PREDICTION_SEASONS)

    womens_train_data = filter_data_dict_by_seasons(full_womens_data,
                                                   seasons_to_include=TRAINING_SEASONS + [VALIDATION_SEASON],
                                                   seasons_to_exclude=PREDICTION_SEASONS)

    # Create separate dictionaries for prediction data (2021-2024)
    mens_prediction_data = filter_data_dict_by_seasons(full_mens_data,
                                                     seasons_to_include=PREDICTION_SEASONS)

    womens_prediction_data = filter_data_dict_by_seasons(full_womens_data,
                                                       seasons_to_include=PREDICTION_SEASONS)

    # Verify no data leakage
    for gender, train_data, pred_data in [("Men's", mens_train_data, mens_prediction_data),
                                         ("Women's", womens_train_data, womens_prediction_data)]:
        if 'df_tourney' in train_data and 'df_tourney' in pred_data:
            train_seasons = train_data['df_tourney']['Season'].unique()
            pred_seasons = pred_data['df_tourney']['Season'].unique()

            overlap = set(train_seasons).intersection(set(pred_seasons))
            if overlap:
                print(f"WARNING: Found overlapping seasons in {gender} data: {overlap}")
            else:
                print(f"{gender} data properly separated with no season overlap")

    return mens_train_data, womens_train_data, mens_prediction_data, womens_prediction_data

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

    is_prediction_season = current_season in PREDICTION_SEASONS

    # Only filter tournament results/games data, not reference data like seeds
    if is_prediction_season:
        # Only filter actual tournament results, keep seed data and other metadata
        if 'WTeamID' in data_df.columns and 'LTeamID' in data_df.columns:
            # Remove tournament results from prediction seasons
            filtered_df = data_df[~data_df[season_col].isin(PREDICTION_SEASONS)]

            # Only print warning if we actually filtered something
            if len(filtered_df) < len(data_df):
                print(f"SAFEGUARD: Removed {len(data_df) - len(filtered_df)} rows of tournament results")

            return filtered_df
        else:
            # Keep all non-tournament data (seeds, etc.) for prediction seasons
            return data_df
    else:
        return data_df