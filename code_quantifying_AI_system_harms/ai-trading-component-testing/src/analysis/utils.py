"""Utilities for combining and annotating raw experiment result CSVs.

When run as a script, merges all CSV files in a given folder into a single
combined results file, adding experiment-type and fixed-asset-type columns.

Usage:
    python src/analysis/utils.py <folder_with_csvs>
"""
import pandas as pd
from typing import List, Optional
import os
import argparse

def combine_experiment_results(results_csv_paths: List[str]) -> pd.DataFrame:

    df_arr = [pd.read_csv(path) for path in results_csv_paths]

    return pd.concat(df_arr)

def add_experiment_column(results_df: pd.DataFrame) -> pd.DataFrame:

    experiment_types = results_df['experiment_name'].str.split("__").str[0].str.split("-").str[1:].str.join("-")
    results_df.insert(2, 'experiment_types', experiment_types)

def add_fixed_asset_type_column(results_df: pd.DataFrame) -> pd.DataFrame:

    def extract_fixed_asset(name):
        parts = name.split("__")
        for part in parts:
            if part.startswith("bullish-"):
                return part.replace("bullish-", "").replace("-","_").lower()
            if part.startswith("bearish-"):
                return part.replace("bearish-", "").replace("-","_").lower()
        return None
    
    bullish_assets = results_df['experiment_name'].apply(extract_fixed_asset)
    results_df.insert(3, 'fixed_asset_type', bullish_assets)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("folder_with_csvs", help="Folder containing csv files to merge")
    args = parser.parse_args()

    csv_files = []
    for dirpath, dirnames, filenames in os.walk(args.folder_with_csvs):
        for file in filenames:
            if 'combined' not in file:
                csv_files.append(os.path.join(dirpath, file))

    combined = combine_experiment_results(csv_files)
    add_experiment_column(combined)
    add_fixed_asset_type_column(combined)

    combined.to_csv("simulation_outputs/llm_response_data/main_run/extension/combined_neutral_bearish_results.csv")
