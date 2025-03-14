import pandas as pd
import numpy as np

def get_data_with_index(data, key_or_column=None, value=None, indexed_suffix='_by_season', fallback_filter=True):
    """
    Helper function to get data using DataFrame indexes if available, falling back to filtering
    
    Can be called in two ways:
    1. get_data_with_index(data_dict, 'df_key', season) - Uses indexed version of DataFrame from dict
    2. get_data_with_index(dataframe, 'column_name', value) - Filters DataFrame directly
    3. get_data_with_index(dataframe, value) - Assumes filtering by 'Season' column
    
    Args:
        data: Dictionary containing dataframes OR a DataFrame
        key_or_column: Either the key for the dataframe in dict, column name, or None
        value: Value to filter by (e.g., a season number or team ID)
        indexed_suffix: Suffix for indexed version (e.g., '_by_season')
        fallback_filter: Whether to fall back to filtering if indexed lookup fails
        
    Returns:
        Filtered DataFrame
    """
    # Handle the call pattern: get_data_with_index(dataframe, season)
    if isinstance(data, pd.DataFrame) and key_or_column is not None and value is None:
        # Assume we're filtering by 'Season'
        value = key_or_column
        key_or_column = 'Season'

    # If the first argument is a DataFrame (not a dict), filter it directly
    if isinstance(data, pd.DataFrame):
        # Check if the DataFrame has an index we can use
        if data.index.name == key_or_column or (isinstance(data.index, pd.MultiIndex) and key_or_column in data.index.names):
            try:
                # Try to use the index
                return data.xs(value, level=key_or_column)
            except (KeyError, TypeError):
                pass
        
        # Fall back to filtering
        if fallback_filter:
            try:
                return data[data[key_or_column] == value]
            except Exception as e:
                print(f"Error filtering DataFrame: {str(e)}")
                return pd.DataFrame()
        return pd.DataFrame()
    
    # Dictionary approach
    if key_or_column is None or value is None:
        print("Error: Both key and value must be provided when data is a dictionary")
        return pd.DataFrame()
        
    indexed_key = f"{key_or_column}{indexed_suffix}"
    
    # Try using the indexed version
    if indexed_key in data and not data[indexed_key].empty:
        try:
            return data[indexed_key].loc[value]
        except (KeyError, TypeError):
            # Key not found in index, fall back to filtering if allowed
            pass
    
    # Fall back to filtering the original DataFrame
    if fallback_filter and key_or_column in data and not data[key_or_column].empty:
        filter_col = indexed_suffix.replace('_by_', '')
        if filter_col == '_season':
            filter_col = 'Season'
        elif filter_col == '_team':
            filter_col = 'TeamID'
        # Apply filter
        try:
            return data[key_or_column][data[key_or_column][filter_col] == value]
        except Exception as e:
            print(f"Error filtering DataFrame from dictionary: {str(e)}")
            return pd.DataFrame()
    
    # Return empty DataFrame if all else fails
    return pd.DataFrame()

def optimize_dataframes_with_indexes(data_dict):
    """
    Adds appropriate indexes to DataFrames for optimized lookups while preserving
    original dataframes for compatibility.
    
    Args:
        data_dict: Dictionary containing the loaded datasets
        
    Returns:
        Dictionary with added indexed versions of DataFrames
    """
    print("Optimizing DataFrames with indexes...")
    
    # Regular Season Results
    if 'df_regular' in data_dict and not data_dict['df_regular'].empty:
        print("  Adding indexes to regular season data...")
        # Create indexed copies for different access patterns
        data_dict['df_regular_by_season'] = data_dict['df_regular'].set_index('Season').sort_index()
        data_dict['df_regular_by_winner'] = data_dict['df_regular'].set_index(['Season', 'WTeamID']).sort_index()
        data_dict['df_regular_by_loser'] = data_dict['df_regular'].set_index(['Season', 'LTeamID']).sort_index()
        if 'MatchupID' in data_dict['df_regular'].columns:
            data_dict['df_regular_by_matchup'] = data_dict['df_regular'].set_index(['Season', 'MatchupID']).sort_index()
    
    # Tournament Results
    if 'df_tourney' in data_dict and not data_dict['df_tourney'].empty:
        print("  Adding indexes to tournament data...")
        data_dict['df_tourney_by_season'] = data_dict['df_tourney'].set_index('Season').sort_index()
        data_dict['df_tourney_by_winner'] = data_dict['df_tourney'].set_index(['Season', 'WTeamID']).sort_index()
        data_dict['df_tourney_by_loser'] = data_dict['df_tourney'].set_index(['Season', 'LTeamID']).sort_index()
        if 'MatchupID' in data_dict['df_tourney'].columns:
            data_dict['df_tourney_by_matchup'] = data_dict['df_tourney'].set_index(['Season', 'MatchupID']).sort_index()
    
    # Seed Data
    if 'df_seed' in data_dict and not data_dict['df_seed'].empty:
        print("  Adding indexes to seed data...")
        data_dict['df_seed_by_season'] = data_dict['df_seed'].set_index('Season').sort_index()
        data_dict['df_seed_by_team'] = data_dict['df_seed'].set_index(['Season', 'TeamID']).sort_index()
    
    # Team Conferences
    if 'df_team_conferences' in data_dict and not data_dict['df_team_conferences'].empty:
        print("  Adding indexes to team conferences data...")
        data_dict['df_team_conferences_by_season'] = data_dict['df_team_conferences'].set_index('Season').sort_index()
        data_dict['df_team_conferences_by_team'] = data_dict['df_team_conferences'].set_index(['Season', 'TeamID']).sort_index()
        data_dict['df_team_conferences_by_conf'] = data_dict['df_team_conferences'].set_index(['Season', 'ConfAbbrev']).sort_index()
    
    # Tournament Slots
    if 'df_tourney_slots' in data_dict and not data_dict['df_tourney_slots'].empty:
        print("  Adding indexes to tournament slots data...")
        data_dict['df_tourney_slots_by_season'] = data_dict['df_tourney_slots'].set_index('Season').sort_index()
        if 'Slot' in data_dict['df_tourney_slots'].columns:
            data_dict['df_tourney_slots_by_slot'] = data_dict['df_tourney_slots'].set_index(['Season', 'Slot']).sort_index()
    
    # Conference Tournament Games
    if 'df_conf_tourney' in data_dict and not data_dict['df_conf_tourney'].empty:
        print("  Adding indexes to conference tournament data...")
        data_dict['df_conf_tourney_by_season'] = data_dict['df_conf_tourney'].set_index('Season').sort_index()
        data_dict['df_conf_tourney_by_conf'] = data_dict['df_conf_tourney'].set_index(['Season', 'ConfAbbrev']).sort_index()
    
    # Coaches Data
    if 'df_coaches' in data_dict and not data_dict['df_coaches'].empty:
        print("  Adding indexes to coaches data...")
        data_dict['df_coaches_by_season'] = data_dict['df_coaches'].set_index('Season').sort_index()
        data_dict['df_coaches_by_team'] = data_dict['df_coaches'].set_index(['Season', 'TeamID']).sort_index()
        data_dict['df_coaches_by_name'] = data_dict['df_coaches'].set_index(['Season', 'CoachName']).sort_index()
    
    print("DataFrame indexing completed.")
    return data_dict


def optimize_feature_dataframes(modeling_data):
    """
    Adds indexes to feature DataFrames created during modeling
    
    Args:
        modeling_data: Dictionary with prepared data
        
    Returns:
        Dictionary with indexed feature DataFrames
    """
    print("Adding indexes to feature DataFrames...")
    
    # Team profiles
    if 'enhanced_team_profiles' in modeling_data and not modeling_data['enhanced_team_profiles'].empty:
        print("  Indexing team profiles...")
        # Create indexed copy rather than modifying in place
        modeling_data['enhanced_team_profiles_by_team'] = modeling_data['enhanced_team_profiles'].set_index(['Season', 'TeamID']).sort_index()
    
    # Tournament history
    if 'tourney_history' in modeling_data and not modeling_data['tourney_history'].empty:
        print("  Indexing tournament history...")
        modeling_data['tourney_history_by_team'] = modeling_data['tourney_history'].set_index(['Season', 'TeamID']).sort_index()
    
    # Round performance
    if 'round_performance' in modeling_data and not modeling_data['round_performance'].empty:
        if 'Season' in modeling_data['round_performance'].columns:
            print("  Indexing round performance...")
            modeling_data['round_performance_by_team'] = modeling_data['round_performance'].set_index(['Season', 'TeamID']).sort_index()
        else:
            print("  Warning: 'Season' column missing in round_performance, skipping indexing")
    
    # Pressure metrics
    if 'pressure_metrics' in modeling_data and not modeling_data['pressure_metrics'].empty:
        if 'Season' in modeling_data['pressure_metrics'].columns:
            print("  Indexing pressure metrics...")
            modeling_data['pressure_metrics_by_team'] = modeling_data['pressure_metrics'].set_index(['Season', 'TeamID']).sort_index()
        else:
            print("  Warning: 'Season' column missing in pressure_metrics, skipping indexing")
    
    # Seed features
    if 'seed_features' in modeling_data and not modeling_data['seed_features'].empty:
        print("  Indexing seed features...")
        modeling_data['seed_features_by_team'] = modeling_data['seed_features'].set_index(['Season', 'TeamID']).sort_index()
    
    # Conference impact
    if 'conf_impact' in modeling_data and modeling_data['conf_impact'] is not None and not modeling_data['conf_impact'].empty:
        print("  Indexing conference impact...")
        modeling_data['conf_impact_by_team'] = modeling_data['conf_impact'].set_index(['Season', 'TeamID']).sort_index()
    
    # Coach metrics
    if 'coach_metrics' in modeling_data and modeling_data['coach_metrics'] is not None and not modeling_data['coach_metrics'].empty:
        # Check if required columns exist
        if 'Season' in modeling_data['coach_metrics'].columns and 'TeamID' in modeling_data['coach_metrics'].columns:
            print("  Indexing coach metrics...")
            modeling_data['coach_metrics_by_team'] = modeling_data['coach_metrics'].set_index(['Season', 'TeamID']).sort_index()
        else:
            print("  Warning: Required columns missing in coach_metrics, skipping indexing")
    
    print("Feature DataFrame indexing completed.")
    return modeling_data


def get_team_data(data_dict, team_id, season, dataset_key, indexed_suffix='_by_team'):
    """
    Specialized helper to get team-specific data using indexed lookups
    
    Args:
        data_dict: Dictionary containing dataframes
        team_id: Team ID to look up
        season: Season to look up
        dataset_key: Base key for the dataframe (e.g., 'enhanced_team_profiles')
        indexed_suffix: Suffix for indexed version
        
    Returns:
        DataFrame or Series with team data
    """
    indexed_key = f"{dataset_key}{indexed_suffix}"
    
    # Try using the indexed version
    if indexed_key in data_dict and not data_dict[indexed_key].empty:
        try:
            return data_dict[indexed_key].loc[(season, team_id)]
        except (KeyError, TypeError):
            # Key not found in index, fall back to filtering
            pass
    
    # Fall back to filtering the original DataFrame
    if dataset_key in data_dict and not data_dict[dataset_key].empty:
        return data_dict[dataset_key][(data_dict[dataset_key]['Season'] == season) & 
                                     (data_dict[dataset_key]['TeamID'] == team_id)]
    
    # Return empty DataFrame if all else fails
    return pd.DataFrame()