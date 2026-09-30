import pandas as pd
import numpy as np
import os

def generate_synthetic_data(num_samples=10000):
    np.random.seed(42)
    
    # Features as per requirements
    weather_conditions = ['Normal', 'Raining', 'Snowing', 'Fog or mist', 'Other', 'Unknown']
    light_conditions = ['Daylight', 'Darkness - lights lit', 'Darkness - no lighting', 'Darkness - lighting unknown']
    road_surface = ['Dry', 'Wet or damp', 'Snow', 'Ice', 'Flood over road']
    vehicle_types = ['Car', 'Motorcycle', 'Bus/Coach', 'Goods vehicle', 'Pedal cycle', 'Other']
    speed_limits = [20, 30, 40, 50, 60, 70]
    time_of_day = ['Morning', 'Afternoon', 'Evening', 'Night']
    
    # Target categories with class imbalance
    # Slight (80%), Serious (15%), Fatal (5%)
    severities = ['Slight', 'Serious', 'Fatal']
    severity_probs = [0.80, 0.15, 0.05]
    
    data = []
    for _ in range(num_samples):
        # Determine severity first to introduce some correlations
        severity = np.random.choice(severities, p=severity_probs)
        
        # Bias features based on severity for a more realistic dataset
        if severity == 'Fatal':
            weather = np.random.choice(weather_conditions, p=[0.2, 0.3, 0.2, 0.2, 0.05, 0.05])
            light = np.random.choice(light_conditions, p=[0.1, 0.2, 0.6, 0.1])
            surface = np.random.choice(road_surface, p=[0.1, 0.3, 0.3, 0.2, 0.1])
            speed = np.random.choice(speed_limits, p=[0.05, 0.05, 0.1, 0.2, 0.3, 0.3])
            vehicle = np.random.choice(vehicle_types, p=[0.3, 0.3, 0.1, 0.2, 0.05, 0.05])
            time = np.random.choice(time_of_day, p=[0.1, 0.1, 0.3, 0.5])
        elif severity == 'Serious':
            weather = np.random.choice(weather_conditions, p=[0.4, 0.3, 0.1, 0.1, 0.05, 0.05])
            light = np.random.choice(light_conditions, p=[0.3, 0.4, 0.2, 0.1])
            surface = np.random.choice(road_surface, p=[0.3, 0.4, 0.1, 0.1, 0.1])
            speed = np.random.choice(speed_limits, p=[0.1, 0.2, 0.3, 0.2, 0.1, 0.1])
            vehicle = np.random.choice(vehicle_types, p=[0.4, 0.2, 0.1, 0.1, 0.1, 0.1])
            time = np.random.choice(time_of_day, p=[0.2, 0.3, 0.3, 0.2])
        else: # Slight
            weather = np.random.choice(weather_conditions, p=[0.7, 0.15, 0.05, 0.05, 0.025, 0.025])
            light = np.random.choice(light_conditions, p=[0.7, 0.2, 0.05, 0.05])
            surface = np.random.choice(road_surface, p=[0.7, 0.2, 0.05, 0.025, 0.025])
            speed = np.random.choice(speed_limits, p=[0.3, 0.4, 0.1, 0.1, 0.05, 0.05])
            vehicle = np.random.choice(vehicle_types, p=[0.6, 0.1, 0.1, 0.1, 0.05, 0.05])
            time = np.random.choice(time_of_day, p=[0.4, 0.4, 0.1, 0.1])
            
        data.append([weather, light, surface, speed, vehicle, time, severity])
        
    df = pd.DataFrame(data, columns=['Weather_Conditions', 'Light_Conditions', 'Road_Surface_Conditions', 
                                     'Speed_Limit', 'Vehicle_Type', 'Time_of_Day', 'Accident_Severity'])
                                     
    # Introduce missing values in weather and road condition fields as per objective
    # 5% missing values
    weather_missing_idx = np.random.choice(df.index, size=int(num_samples * 0.05), replace=False)
    road_missing_idx = np.random.choice(df.index, size=int(num_samples * 0.05), replace=False)
    
    df.loc[weather_missing_idx, 'Weather_Conditions'] = np.nan
    df.loc[road_missing_idx, 'Road_Surface_Conditions'] = np.nan
    
    return df

if __name__ == "__main__":
    print("Generating synthetic dataset...")
    df = generate_synthetic_data(10000)
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/road_accidents.csv', index=False)
    print("Dataset generated and saved to data/road_accidents.csv")
    print("\nDataset Info:")
    print(df.info())
    print("\nSeverity Distribution:")
    print(df['Accident_Severity'].value_counts(normalize=True))
