import pandas as pd 
import numpy as np
import os

def filter_time(df, season=2015):
    """
    Filter dataframe to include only data from specified season onward

    Args:
        df: DataFrame to filter
        season: Starting season to include

    Returns:
        Filtered DataFrame
    """
    if 'Season' in df.columns:
        return df[df['Season'] >= season]
    return df


def extract_seed_number(seed_str):
    """
    Extract numerical seed from seed string (e.g. 'W01' -> 1)

    Args:
        seed_str: Seed string

    Returns:
        Integer seed number
    """
    if isinstance(seed_str, str) and len(seed_str) >= 2:
        return int(seed_str[1:3])
    return None

def load_mens_data(starting_season=2015, data_dir=None):
    """
    Load only men's NCAA basketball datasets filtered by starting season

    Args:
        starting_season (int): The starting season to include (default: 2015)
        data_dir (str): Directory containing data files (default: None)

    Returns:
        dict: A dictionary containing the loaded dataframes with standardized keys
    """
    import os
    import pandas as pd

    # Set default data directory if none provided
    if data_dir is None:
        # Try to find the data directory relative to the project root
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        data_dir = os.path.join(project_root, 'data')
    
    # List of men's filenames to load
    mens_files = [
        'MNCAATourneyDetailedResults.csv',
        'MNCAATourneySeeds.csv',
        'MRegularSeasonDetailedResults.csv',
        'MTeamCoaches.csv',
        'MTeamConferences.csv',
        'MTeams.csv',
        'MConferenceTourneyGames.csv',
        'MNCAATourneySlots.csv',
        'MNCAATourneySeedRoundSlots.csv'
    ]

    # Dictionary to store dataframes
    data_dict = {}

    # Load each file
    for file in mens_files:
        file_path = os.path.join(data_dir, file)
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path, encoding='cp1252')
                df = filter_time(df, starting_season)

                # Standardize the variable name
                var_name = ("df_" + file.replace('M', '')
                                    .replace('NCAAT', 't')
                                    .replace('.csv', '')
                                    .replace('CSV', '')
                                    .lower())
                data_dict[var_name] = df
                print(f"Loaded {file}: {len(df)} rows")
            except Exception as e:
                print(f"Error loading {file}: {e}")
        else:
            print(f"File not found: {file_path}")

    # Create alias names for pipeline compatibility
    alias_mapping = {
        'df_tourneyseeds.csv': 'df_seed',
        'df_tourneydetailedresults.csv': 'df_tourney',
        'df_regularseasondetailedresults.csv': 'df_regular',
        'df_tourneyslots.csv': 'df_tourney_slots',
        'df_conferencetourneygames.csv': 'df_conf_tourney',
        'df_teamconferences.csv': 'df_team_conferences',
        'df_teamcoaches.csv': 'df_coaches',
        'df_tourneyseedroundslots.csv': 'df_seed_round_slots'
    }
    for old_name, new_name in alias_mapping.items():
        old_key = old_name.replace('.csv', '')
        if old_key in data_dict:
            data_dict[new_name] = data_dict[old_key]

    # For tournament and regular season datasets, create a MatchupID
    for key in ['df_tourney', 'df_regular']:
        if key in data_dict:
            df = data_dict[key]
            df['MatchupID'] = df.apply(lambda row: f"{row['Season']}_{min(row['WTeamID'], row['LTeamID'])}_{max(row['WTeamID'], row['LTeamID'])}",
                                       axis=1)
            data_dict[key] = df

    print(f"Successfully loaded {len(data_dict)} men's datasets from season {starting_season} onwards.")
    return data_dict

def load_womens_data(starting_season=2015, data_dir=None):
    """
    Load only women's NCAA basketball datasets filtered by starting season

    Args:
        starting_season (int): The starting season to include (default: 2015)
        data_dir (str): Directory containing data files (default: None)

    Returns:
        dict: A dictionary containing the loaded dataframes with standardized keys
    """

    # Set default data directory if none provided
    if data_dir is None:
        # Try to find the data directory relative to the project root
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        data_dir = os.path.join(project_root, 'data')

    # List of women's filenames to load
    womens_files = [
        'WNCAATourneyDetailedResults.csv',
        'WNCAATourneySeeds.csv',
        'WRegularSeasonDetailedResults.csv',
        'WTeamConferences.csv',
        'WTeams.csv',
        'WConferenceTourneyGames.csv',
        'WNCAATourneySlots.csv'
    ]

    # Dictionary to store dataframes
    data_dict = {}

    # Load each file
    for file in womens_files:
        file_path = os.path.join(data_dir, file)
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path, encoding='cp1252')
                df = filter_time(df, starting_season)

                # Standardize the variable name
                var_name = ("df_" + file.replace('W', '')
                                     .replace('NCAAT', 't')
                                     .replace('.csv', '')
                                     .replace('CSV', '')
                                     .lower())
                data_dict[var_name] = df
                print(f"Loaded {file}: {len(df)} rows")
            except Exception as e:
                print(f"Error loading {file}: {e}")
        else:
            print(f"File not found: {file}")

    # Create alias names for pipeline compatibility
    alias_mapping = {
        'df_tourneyseeds.csv': 'df_seed',
        'df_tourneydetailedresults.csv': 'df_tourney',
        'df_regularseasondetailedresults.csv': 'df_regular',
        'df_tourneyslots.csv': 'df_tourney_slots',
        'df_conferencetourneygames.csv': 'df_conf_tourney',
        'df_teamconferences.csv': 'df_team_conferences'
    }
    for old_name, new_name in alias_mapping.items():
        old_key = old_name.replace('.csv', '')
        if old_key in data_dict:
            data_dict[new_name] = data_dict[old_key]

    # For tournament and regular season datasets, create a MatchupID
    for key in ['df_tourney', 'df_regular']:
        if key in data_dict:
            df = data_dict[key]
            df['MatchupID'] = df.apply(lambda row: f"{row['Season']}_{min(row['WTeamID'], row['LTeamID'])}_{max(row['WTeamID'], row['LTeamID'])}",
                                       axis=1)
            data_dict[key] = df

    print(f"Successfully loaded {len(data_dict)} women's datasets from season {starting_season} onwards.")
    return data_dict

def filter_data_dict_by_seasons(data_dict, seasons_to_include, seasons_to_exclude=None):
    """
    Filter all dataframes in a data dictionary by season.

    Args:
        data_dict: Dictionary containing dataframes
        seasons_to_include: List of seasons to include
        seasons_to_exclude: List of seasons to explicitly exclude

    Returns:
        Filtered data dictionary
    """
    filtered_dict = {}

    for key, df in data_dict.items():
        # Copy the original dataframe to avoid modifying it
        filtered_df = df.copy()

        # Apply season filtering if the dataframe has a Season column
        if 'Season' in filtered_df.columns:
            # Include specified seasons
            if seasons_to_include:
                filtered_df = filtered_df[filtered_df['Season'].isin(seasons_to_include)]

            # Exclude specified seasons
            if seasons_to_exclude:
                filtered_df = filtered_df[~filtered_df['Season'].isin(seasons_to_exclude)]

        filtered_dict[key] = filtered_df

    return filtered_dict
