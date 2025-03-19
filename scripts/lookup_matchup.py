#!/usr/bin/env python
"""
NCAA March Madness Team Matchup Predictor Lookup Tool

This script allows you to quickly look up prediction probabilities for any team matchup 
by entering team names. It handles various spellings of team names and displays the
prediction from your submission file.

Usage:
  python lookup_matchup.py [--predictions PREDICTIONS_FILE] [--year YEAR]
  
Example:
  python lookup_matchup.py --predictions submission_2025.csv --year 2025
  
Then follow the prompts to enter team names.
"""

import os
import sys
import pandas as pd
import argparse
from difflib import get_close_matches

def get_project_root():
    """Determine the project root directory"""
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # If the script is in a 'scripts' directory, go up one level
    if os.path.basename(script_dir) == 'scripts':
        return os.path.dirname(script_dir)
    
    return script_dir

def load_team_spellings(mens_file='MTeamSpellings.csv', womens_file='WTeamSpellings.csv'):
    """Load team spellings data from both men's and women's files"""
    try:
        # Get project root and data directory
        project_root = get_project_root()
        data_dir = os.path.join(project_root, 'data')
        
        # Full paths to the team spellings files
        mens_path = os.path.join(data_dir, mens_file)
        womens_path = os.path.join(data_dir, womens_file)
        
        print(f"Loading team spellings from:")
        print(f"  - {mens_path}")
        print(f"  - {womens_path}")
        
        mens_spellings = pd.read_csv(mens_path)
        womens_spellings = pd.read_csv(womens_path)
        
        # Create dictionaries for quick lookup (spelling -> TeamID)
        mens_dict = {}
        for _, row in mens_spellings.iterrows():
            mens_dict[row['TeamNameSpelling'].lower()] = row['TeamID']
            
        womens_dict = {}
        for _, row in womens_spellings.iterrows():
            womens_dict[row['TeamNameSpelling'].lower()] = row['TeamID']
            
        # Create reverse dictionaries for TeamID -> name lookups
        mens_id_to_name = {}
        womens_id_to_name = {}
        
        for name, team_id in mens_dict.items():
            if team_id not in mens_id_to_name:
                mens_id_to_name[team_id] = name
                
        for name, team_id in womens_dict.items():
            if team_id not in womens_id_to_name:
                womens_id_to_name[team_id] = name
        
        # Create lists of all team name spellings for fuzzy matching
        mens_names = list(mens_dict.keys())
        womens_names = list(womens_dict.keys())
        
        return {
            'mens_dict': mens_dict,
            'womens_dict': womens_dict,
            'mens_names': mens_names,
            'womens_names': womens_names,
            'mens_id_to_name': mens_id_to_name,
            'womens_id_to_name': womens_id_to_name
        }
    except FileNotFoundError as e:
        print(f"Error: Could not find team spellings file. {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading team spellings data: {e}")
        sys.exit(1)

def load_predictions(predictions_file='submission_2025.csv'):
    """Load the predictions file"""
    try:
        # If the predictions file path is not absolute, look for it in the project root
        if not os.path.isabs(predictions_file):
            project_root = get_project_root()
            full_path = os.path.join(project_root, predictions_file)
            if os.path.exists(full_path):
                predictions_file = full_path
        
        print(f"Loading predictions from: {predictions_file}")
        predictions = pd.read_csv(predictions_file)
        
        # Create a dictionary for fast lookup
        pred_dict = {}
        for _, row in predictions.iterrows():
            pred_dict[row['ID']] = row['Pred']
            
        return pred_dict
    except FileNotFoundError:
        print(f"Error: Could not find predictions file '{predictions_file}'")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Please make sure the file exists or provide the full path")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading predictions: {e}")
        sys.exit(1)

def lookup_team_id(team_name, spellings_data, gender='both'):
    """Look up a team ID from a team name, with fuzzy matching for suggestions"""
    team_name = team_name.lower().strip()
    
    # Determine which dictionaries to use based on gender
    if gender == 'mens' or gender == 'm':
        dicts_to_check = [('mens', spellings_data['mens_dict'], spellings_data['mens_names'])]
    elif gender == 'womens' or gender == 'w':
        dicts_to_check = [('womens', spellings_data['womens_dict'], spellings_data['womens_names'])]
    else:
        dicts_to_check = [
            ('mens', spellings_data['mens_dict'], spellings_data['mens_names']),
            ('womens', spellings_data['womens_dict'], spellings_data['womens_names'])
        ]
    
    # Check each dictionary
    for gender_label, name_dict, all_names in dicts_to_check:
        if team_name in name_dict:
            return name_dict[team_name], gender_label
        
        # Try fuzzy matching if exact match fails
        close_matches = get_close_matches(team_name, all_names, n=3, cutoff=0.6)
        
        if close_matches:
            print(f"\nTeam '{team_name}' not found exactly. Did you mean one of these {gender_label} teams?")
            for i, match in enumerate(close_matches, 1):
                print(f"  {i}. {match}")
            
            choice = input("Enter the number of your choice (or press Enter to skip): ")
            if choice.isdigit() and 1 <= int(choice) <= len(close_matches):
                selected = close_matches[int(choice) - 1]
                return name_dict[selected], gender_label
    
    return None, None

def create_matchup_id(team1_id, team2_id, year=2025):
    """Create a matchup ID with the lower team ID first"""
    lower_id = min(team1_id, team2_id)
    higher_id = max(team1_id, team2_id)
    return f"{year}_{lower_id}_{higher_id}"

def interpret_result(probability, team1_id, team2_id, team1_name, team2_name, matchup_id):
    """Interpret the prediction probability for user-friendly display"""
    # Extract the first team ID from the matchup ID
    first_team_id = int(matchup_id.split('_')[1])
    
    # Determine if team1 is the first team in the matchup ID
    if team1_id == first_team_id:
        win_prob = probability * 100
        team1_is_first = True
    else:
        win_prob = (1 - probability) * 100
        team1_is_first = False
    
    result = {
        'team1_name': team1_name,
        'team2_name': team2_name,
        'team1_id': team1_id,
        'team2_id': team2_id,
        'team1_win_prob': win_prob,
        'team2_win_prob': 100 - win_prob,
        'matchup_id': matchup_id,
        'raw_prediction': probability,
        'team1_is_first': team1_is_first
    }
    
    return result

def display_result(result):
    """Format and display the matchup prediction result"""
    print("\n" + "="*60)
    print(f"MATCHUP: {result['team1_name'].title()} vs {result['team2_name'].title()}")
    print(f"Team IDs: {result['team1_id']} vs {result['team2_id']}")
    print(f"Matchup ID: {result['matchup_id']}")
    print("-"*60)
    print(f"Win Probability for {result['team1_name'].title()}: {result['team1_win_prob']:.1f}%")
    print(f"Win Probability for {result['team2_name'].title()}: {result['team2_win_prob']:.1f}%")
    print("-"*60)
    
    # Explain how prediction is stored
    if result['team1_is_first']:
        first_team = result['team1_name'].title()
        raw_prob = result['raw_prediction']
    else:
        first_team = result['team2_name'].title()
        raw_prob = result['raw_prediction']
        
    print(f"Note: Raw prediction ({raw_prob:.4f}) represents the probability that")
    print(f"      {first_team} (the team with the lower ID) wins the matchup.")
    print("="*60 + "\n")

def main():
    parser = argparse.ArgumentParser(description='NCAA March Madness Matchup Predictor Lookup Tool')
    parser.add_argument('--predictions', default='submission_2025.csv', 
                        help='Path to predictions CSV file')
    parser.add_argument('--year', default=2025, type=int,
                        help='Tournament year (default: 2025)')
    parser.add_argument('--men-spellings', default='MTeamSpellings.csv',
                        help='Filename of men\'s team spellings file in data directory')
    parser.add_argument('--women-spellings', default='WTeamSpellings.csv',
                        help='Filename of women\'s team spellings file in data directory')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Looking up team data and predictions...")
    spellings_data = load_team_spellings(args.men_spellings, args.women_spellings)
    predictions = load_predictions(args.predictions)
    
    print(f"\nLoaded {len(spellings_data['mens_dict'])} men's team spellings and " 
          f"{len(spellings_data['womens_dict'])} women's team spellings")
    print(f"Loaded {len(predictions)} matchup predictions for {args.year}\n")
    
    # Interactive loop
    while True:
        print("\n" + "="*60)
        print("NCAA MARCH MADNESS MATCHUP PREDICTOR")
        print("="*60)
        print("Enter two team names to see their predicted matchup result")
        print("Type 'q' to quit, 'h' for help")
        print("-"*60)
        
        # Get team 1
        team1_input = input("\nEnter first team name: ")
        if team1_input.lower() in ['q', 'quit', 'exit']:
            break
        if team1_input.lower() in ['h', 'help']:
            print("\nHELP: Enter any team name or part of a name. The system will")
            print("      suggest matches if an exact match isn't found.")
            print("      For gender-specific lookup, prefix with 'm:' or 'w:'")
            print("      Example: 'm:Duke' for men's Duke team")
            continue
            
        # Check for gender prefix
        gender = 'both'
        if team1_input.lower().startswith('m:'):
            gender = 'mens'
            team1_input = team1_input[2:].strip()
        elif team1_input.lower().startswith('w:'):
            gender = 'womens'
            team1_input = team1_input[2:].strip()
            
        team1_id, team1_gender = lookup_team_id(team1_input, spellings_data, gender)
        
        if team1_id is None:
            print(f"Could not find a match for '{team1_input}'. Please try again.")
            continue
            
        # Remember the gender for second team suggestion
        gender = team1_gender
        
        # Get team 2
        team2_input = input("\nEnter second team name: ")
        if team2_input.lower() in ['q', 'quit', 'exit']:
            break
            
        # Check for gender prefix (overrides first team's gender)
        if team2_input.lower().startswith('m:'):
            gender = 'mens'
            team2_input = team2_input[2:].strip()
        elif team2_input.lower().startswith('w:'):
            gender = 'womens'
            team2_input = team2_input[2:].strip()
            
        team2_id, team2_gender = lookup_team_id(team2_input, spellings_data, gender)
        
        if team2_id is None:
            print(f"Could not find a match for '{team2_input}'. Please try again.")
            continue
            
        # Verify teams are from same gender tournament
        if team1_gender != team2_gender:
            print(f"Error: Cannot create matchup between {team1_gender} and {team2_gender} teams")
            continue
            
        # Create matchup ID and lookup prediction
        matchup_id = create_matchup_id(team1_id, team2_id, args.year)
        
        if matchup_id in predictions:
            prediction = predictions[matchup_id]
            
            # Get proper team names for display (use the original input as a default)
            team1_name = team1_input
            team2_name = team2_input
            
            # Try to get a cleaner name from our reverse lookup
            id_to_name = spellings_data[f'{team1_gender}_id_to_name']
            if team1_id in id_to_name:
                team1_name = id_to_name[team1_id]
            if team2_id in id_to_name:
                team2_name = id_to_name[team2_id]
                
            # Interpret and display result
            result = interpret_result(prediction, team1_id, team2_id, team1_name, team2_name, matchup_id)
            display_result(result)
        else:
            print(f"No prediction found for matchup {matchup_id}")
            print(f"This is strange since we should have predictions for all possible matchups.")
            print(f"Double-check that you're using the correct predictions file.")

if __name__ == "__main__":
    main()