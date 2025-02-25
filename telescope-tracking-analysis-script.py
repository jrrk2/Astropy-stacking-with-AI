import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def parse_tracking_data(file_path):
    """
    Parse tracking data from the observation log file
    """
    data = []
    
    # Regex patterns to extract information
    fits_pattern = re.compile(r'FITS:\s+(.+\.fits)')
    altaz_pattern = re.compile(r'ALT/AZ:\s+([\d.]+)°,\s+([\d.]+)°')
    timestamp_pattern = re.compile(r'light_(\d{8}_\d{6})')
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find all matches
    fits_files = fits_pattern.findall(content)
    altaz_matches = altaz_pattern.findall(content)
    timestamp_matches = timestamp_pattern.findall(content)
    
    # Ensure we have matching lengths
    if len(fits_files) == len(altaz_matches) == len(timestamp_matches):
        for filename, (alt, az), timestamp_str in zip(fits_files, altaz_matches, timestamp_matches):
            # Parse timestamp
            timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
            
            data.append({
                'filename': filename,
                'timestamp': timestamp,
                'alt': float(alt),
                'az': float(az)
            })
    
    # Convert to DataFrame and sort by timestamp
    df = pd.DataFrame(data)
    df = df.sort_values('timestamp')
    
    return df

def analyze_tracking(df):
    """
    Analyze telescope tracking performance
    """
    # Calculate time from first observation
    df['time_delta'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds() / 60

    # Fit linear models for Alt and Az
    alt_model = np.polyfit(df['time_delta'], df['alt'], 1)
    az_model = np.polyfit(df['time_delta'], df['az'], 1)

    # Calculate ideal positions based on linear models
    df['ideal_alt'] = np.polyval(alt_model, df['time_delta'])
    df['ideal_az'] = np.polyval(az_model, df['time_delta'])

    # Calculate tracking errors
    df['alt_error'] = df['alt'] - df['ideal_alt']
    df['az_error'] = df['az'] - df['ideal_az']

    # Plotting
    plt.figure(figsize=(15, 10))

    # Altitude subplot
    plt.subplot(2, 2, 1)
    plt.scatter(df['time_delta'], df['alt'], label='Actual Alt', color='blue', alpha=0.7)
    plt.plot(df['time_delta'], df['ideal_alt'], label='Ideal Alt', color='red', linestyle='--')
    plt.title('Altitude Tracking')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Altitude (degrees)')
    plt.legend()

    # Azimuth subplot
    plt.subplot(2, 2, 2)
    plt.scatter(df['time_delta'], df['az'], label='Actual Az', color='green', alpha=0.7)
    plt.plot(df['time_delta'], df['ideal_az'], label='Ideal Az', color='red', linestyle='--')
    plt.title('Azimuth Tracking')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Azimuth (degrees)')
    plt.legend()

    # Altitude Error subplot
    plt.subplot(2, 2, 3)
    plt.scatter(df['time_delta'], df['alt_error'], color='blue')
    plt.title('Altitude Tracking Error')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Error (degrees)')
    plt.axhline(y=0, color='r', linestyle='--')

    # Azimuth Error subplot
    plt.subplot(2, 2, 4)
    plt.scatter(df['time_delta'], df['az_error'], color='green')
    plt.title('Azimuth Tracking Error')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Error (degrees)')
    plt.axhline(y=0, color='r', linestyle='--')

    plt.tight_layout()
    plt.savefig('telescope_tracking_analysis.png')

    # Print statistical summary
    print("Altitude Tracking Error Statistics:")
    print(df['alt_error'].describe())
    print("\nAzimuth Tracking Error Statistics:")
    print(df['az_error'].describe())

    return df

# Main execution
df = parse_tracking_data('paste.txt')
analyzed_df = analyze_tracking(df)
