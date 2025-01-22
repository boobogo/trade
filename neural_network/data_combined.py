import pandas as pd

def get_data(*dataframes):
    """
    Concatenate an indefinite number of DataFrames and export the combined DataFrame as a CSV file.
    
    Parameters:
    - *dataframes: Indefinite number of DataFrame names to concatenate.
    
    The function will name the file based on the time frames used.
    """
    # Define the file paths
    file_paths = {
        'df_1min': 'D:/boobogo/coding/trade/data/us100/data_engineered/US_TECH100_1min_indicators.csv',
        'df_3min': 'D:/boobogo/coding/trade/data/us100/data_engineered/US_TECH100_3min_indicators.csv',
        'df_5min': 'D:/boobogo/coding/trade/data/us100/data_engineered/US_TECH100_5min_indicators.csv',
        'df_15min': 'D:/boobogo/coding/trade/data/us100/data_engineered/US_TECH100_15min_indicators.csv',
        'df_30min': 'D:/boobogo/coding/trade/data/us100/data_engineered/US_TECH100_30min_indicators.csv',
        'df_1hour': 'D:/boobogo/coding/trade/data/us100/data_engineered/US_TECH100_1hour_indicators.csv',
        'df_4hour': 'D:/boobogo/coding/trade/data/us100/data_engineered/US_TECH100_4hour_indicators.csv',
        'df_12hour': 'D:/boobogo/coding/trade/data/us100/data_engineered/US_TECH100_12hour_indicators.csv',
        'df_1d': 'D:/boobogo/coding/trade/data/us100/data_engineered/US_TECH100_1d_indicators.csv',
        'df_3d': 'D:/boobogo/coding/trade/data/us100/data_engineered/US_TECH100_3d_indicators.csv',
        'df_1w': 'D:/boobogo/coding/trade/data/us100/data_engineered/US_TECH100_1w_indicators.csv'
    }

    # Load the specified DataFrames
    dfs = [pd.read_csv(file_paths[df_name], parse_dates=True, index_col='datetime') for df_name in dataframes]

    # Concatenate the DataFrames
    df_combined = pd.concat(dfs, axis=1)

    # Fill missing values with the forward fill method
    df_combined.ffill(inplace=True)

    # Drop the leading rows with missing values
    df_combined.dropna(inplace=True)
    return df_combined


# Example usage
if __name__ == "__main__":
    # Example usage of the function
    print(get_data('df_4hour', 'df_1d','df_1w'))