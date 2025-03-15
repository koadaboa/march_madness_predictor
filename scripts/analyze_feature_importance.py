#!/usr/bin/env python
# Script to analyze and compare feature importances between men's and women's models
import os
import sys
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Add the package to path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    print("Analyzing Feature Importance Differences Between Men's and Women's Models")
    
    # Load model metadata
    with open('models/mens_metadata.pkl', 'rb') as f:
        mens_metadata = pickle.load(f)

    with open('models/womens_metadata.pkl', 'rb') as f:
        womens_metadata = pickle.load(f)
    
    # Load trained models
    with open('models/mens_model.pkl', 'rb') as f:
        mens_model = pickle.load(f)
        
    with open('models/womens_model.pkl', 'rb') as f:
        womens_model = pickle.load(f)
    
    # Extract feature importances from XGBoost component
    mens_importances = mens_model.named_estimators_['xgb'].feature_importances_
    womens_importances = womens_model.named_estimators_['xgb'].feature_importances_
    
    # Make sure the feature lists are the same length
    min_length = min(len(mens_metadata['feature_cols']), len(womens_importances))
    
    # Create comparison dataframe
    importance_comparison = pd.DataFrame({
        'Feature': mens_metadata['feature_cols'][:min_length],
        'Mens_Importance': mens_importances[:min_length],
        'Womens_Importance': womens_importances[:min_length],
        'Difference': mens_importances[:min_length] - womens_importances[:min_length]
    }).sort_values('Difference', ascending=False)
    
    # Display features with largest differences
    print("\nFeatures MORE important in MEN'S model:")
    print(importance_comparison.head(10).to_string())
    
    print("\nFeatures MORE important in WOMEN'S model:")
    print(importance_comparison.tail(10).to_string())
    
    # Calculate absolute difference to find features with biggest overall difference
    importance_comparison['Abs_Difference'] = np.abs(importance_comparison['Difference'])
    
    print("\nFeatures with LARGEST overall importance difference:")
    print(importance_comparison.sort_values('Abs_Difference', ascending=False).head(10).to_string())
    
    # Also look at top features by importance for each model
    print("\nTop 10 features in MEN'S model:")
    mens_top = pd.DataFrame({
        'Feature': mens_metadata['feature_cols'][:min_length],
        'Importance': mens_importances[:min_length]
    }).sort_values('Importance', ascending=False).head(10)
    print(mens_top.to_string())
    
    print("\nTop 10 features in WOMEN'S model:")
    womens_top = pd.DataFrame({
        'Feature': mens_metadata['feature_cols'][:min_length],
        'Importance': womens_importances[:min_length]
    }).sort_values('Importance', ascending=False).head(10)
    print(womens_top.to_string())
    
    # Optional: Create visualization
    plt.figure(figsize=(12, 8))
    merged_top_features = pd.concat([
        mens_top[['Feature']].head(10), 
        womens_top[['Feature']].head(10)
    ]).drop_duplicates()
    
    important_features = merged_top_features['Feature'].tolist()
    
    compare_df = importance_comparison[importance_comparison['Feature'].isin(important_features)]
    compare_df = compare_df.sort_values('Abs_Difference', ascending=False)
    
    x = np.arange(len(compare_df))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 10))
    mens_bars = ax.barh(x - width/2, compare_df['Mens_Importance'], width, label="Men's Model")
    womens_bars = ax.barh(x + width/2, compare_df['Womens_Importance'], width, label="Women's Model")
    
    ax.set_yticks(x)
    ax.set_yticklabels(compare_df['Feature'])
    ax.legend()
    ax.set_title('Feature Importance Comparison Between Men\'s and Women\'s Models')
    
    plt.tight_layout()
    plt.savefig('feature_importance_comparison.png')
    print("\nVisualization saved as 'feature_importance_comparison.png'")

if __name__ == "__main__":
    main()