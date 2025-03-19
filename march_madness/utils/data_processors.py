import os
import pandas as pd
import numpy as np
import pickle
from ..data.loaders import filter_data_dict_by_seasons
from ..data.processors import filter_reg_season
from ..features.team_features import (create_team_season_profiles, calculate_momentum_features, 
    enhance_team_metrics, calculate_coach_features, calculate_strength_of_schedule, 
    calculate_womens_specific_features)
from ..features.tournament import (calculate_tournament_history, calculate_conference_strength,
    calculate_expected_round_features, create_seed_trend_features, calculate_conference_tournament_impact, 
    calculate_coach_tournament_metrics)
from ..features.matchup import (create_seed_based_features, create_seed_based_pressure_metrics, 
    create_seed_based_trend_features, calculate_seed_based_probability, create_tournament_prediction_dataset)
from ..utils.data_access import optimize_feature_dataframes, get_data_with_index

def prepare_modeling_data(data_dict, gender, starting_season, current_season, seasons_to_process, cache_dir=None):
    """
    Performs feature engineering and creates prediction datasets
    
    Args:
        data_dict: Dictionary containing the loaded datasets
        gender: 'men' or 'women' to identify which dataset is being processed
        starting_season: First season to include in analysis
        current_season: Current season for prediction
        seasons_to_process: List of seasons to process
        cache_dir: Directory to save processed data (optional)
        
    Returns:
        Dictionary with all the prepared data needed for modeling
    """
    print(f"\n=== Preparing modeling data for {gender}'s NCAA Basketball Tournament ===\n")
    
    # Check if cached data exists and should be used
    if cache_dir is not None:
        cache_file = os.path.join(cache_dir, f"{gender}_modeling_data.pkl")
        if os.path.exists(cache_file):
            print(f"Loading cached modeling data from {cache_file}")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
    
    # Add debugging for input data
    if 'df_tourney' in data_dict:
        # Use:
        if 'df_tourney_by_season' in data_dict:
            seasons = sorted(data_dict['df_tourney_by_season'].index.unique())
        elif 'df_tourney' in data_dict:
            seasons = sorted(data_dict['df_tourney']['Season'].unique())
        else:
            seasons = []
        print(f"DEBUG: {gender} tourney data contains seasons:", seasons)
        for season in seasons_to_process:
            games = get_data_with_index(data_dict['df_tourney'], 'Season', season, indexed_suffix='_by_season')
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

    # Create MatchupID for tournament data if not already present
    if 'MatchupID' not in df_tourney.columns and not df_tourney.empty:
        df_tourney['MatchupID'] = df_tourney.apply(
            lambda row: f"{row['Season']}_{min(row['WTeamID'], row['LTeamID'])}_{max(row['WTeamID'], row['LTeamID'])}",
            axis=1
        )

    # Determine tournament days by season to avoid data leakage
    tournament_days = {}
    for season in df_tourney['Season'].unique():
        season_tourney = get_data_with_index(df_tourney, 'Season', season, indexed_suffix='_by_season')
        tournament_days[season] = season_tourney['DayNum'].unique()

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

    if gender == "women's":
        print("\n1a. Adding women's-specific features...")
        womens_features = calculate_womens_specific_features(all_team_games, team_profiles)
        if not womens_features.empty:
            team_profiles = team_profiles.merge(womens_features, on=['Season', 'TeamID'], how='left')
            print(f"Added women's-specific features to {len(team_profiles)} team profiles")
        else:
            print("No women's-specific features were generated")

    print("\n7. Enhancing team metrics...")
    enhanced_team_games, enhanced_team_profiles = enhance_team_metrics(
        all_team_games,
        team_profiles,
        filtered_reg
    )
    print(f"Enhanced metrics for {len(enhanced_team_profiles)} team-seasons")

    print("\n2. Calculating strength of schedule...")
    sos_data = calculate_strength_of_schedule(
        filtered_reg,
        enhanced_team_profiles,
        current_season=current_season,
        tournament_days=tournament_days
    )
    print(f"Calculated SOS for {len(sos_data)} team-seasons")

    print("\n3. Calculating advanced momentum features...")
    momentum_data = calculate_momentum_features(
        all_team_games,
        current_season=current_season,
        tournament_days=tournament_days,
        team_profiles=enhanced_team_profiles,
        sos_data=sos_data
    )
    print(f"Generated advanced momentum features for {len(momentum_data)} team-seasons")

    # Coach features
    if not df_coaches.empty:
        print("\n4. Calculating coach features...")
        # SAFEGUARD: Use safe_tourney_df instead of df_tourney
        coach_features = calculate_coach_features(df_coaches, safe_tourney_df)
        print(f"Generated coach features for {len(coach_features)} team-seasons")
    else:
        coach_features = pd.DataFrame(columns=['Season', 'TeamID', 'CoachName', 'CoachYearsExp', 'CoachTourneyExp', 'CoachChampionships'])
        print("No coach data available, using empty coach features")

    print("\n5. Calculating tournament history...")
    # SAFEGUARD: Use safe_tourney_df instead of df_tourney
    tourney_history = calculate_tournament_history(safe_tourney_df, current_season=current_season)
    print(f"Calculated tournament history for {len(tourney_history)} team-seasons")

    print("\n6. Calculating conference strength...")
    # SAFEGUARD: Use safe_tourney_df instead of df_tourney
    conf_strength = calculate_conference_strength(df_team_conferences, safe_tourney_df, df_seed, current_season=current_season)
    print(f"Calculated conference strength for {len(conf_strength)} conference-seasons")

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
        season_team_profiles = get_data_with_index(enhanced_team_profiles, 'Season', season, indexed_suffix='_by_season')
        all_teams = season_team_profiles['TeamID'].unique()
        
        # Get tournament teams
        season_seed_data = get_data_with_index(df_seed, 'Season', season, indexed_suffix='_by_season')
        
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

    # Add indexes to feature DataFrames
    return_data = optimize_feature_dataframes(return_data)
    
    # Cache the results if a cache directory is provided
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"{gender}_modeling_data_2025.pkl")
        print(f"Caching modeling data to {cache_file}")
        with open(cache_file, 'wb') as f:
            pickle.dump(return_data, f)
    
    return return_data

def load_or_prepare_modeling_data(data_dict, gender, starting_season, current_season, 
                                  seasons_to_process, cache_dir, force_prepare=False):
    """Load cached modeling data or prepare it if not available"""
    if not force_prepare and cache_dir is not None:
        cache_file = os.path.join(cache_dir, f"{gender}_modeling_data.pkl")
        if os.path.exists(cache_file):
            print(f"Loading cached modeling data from {cache_file}")
            try:
                with open(cache_file, 'rb') as f:
                    modeling_data = pickle.load(f)
                
                # Ensure tournament data is available
                if 'tourney_data' not in modeling_data or len(modeling_data['tourney_data']) == 0:
                    print("WARNING: Cached data doesn't contain tournament results. Loading raw data.")
                    if data_dict is None:
                        print("ERROR: Need to load raw data but data_dict is None. Run without --skip-data-load.")
                        return None
                    
                    # Add tournament data from raw data
                    if gender == "men's":
                        modeling_data['tourney_data'] = data_dict.get('df_tourney', pd.DataFrame())
                    else:
                        modeling_data['tourney_data'] = data_dict.get('df_tourney', pd.DataFrame())
                
                return modeling_data
            except Exception as e:
                print(f"Error loading cached data: {e}")
                print("Preparing data from scratch...")