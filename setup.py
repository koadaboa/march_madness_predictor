from setuptools import setup, find_packages

setup(
    name="march_madness_predictor",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "xgboost",
        "lightgbm",
        "catboost",
        "optuna",
        "imbalanced-learn",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A package for predicting NCAA March Madness tournament outcomes",
)