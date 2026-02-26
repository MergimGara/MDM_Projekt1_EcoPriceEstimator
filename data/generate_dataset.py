import pandas as pd
import numpy as np
from pathlib import Path

# Resolve paths relative to this file
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = BASE_DIR / "dataset.csv"

def generate_car_data(n_samples=2000):
    np.random.seed(42)
    brands = ['Toyota', 'VW', 'BMW', 'Ford', 'Tesla', 'Audi']
    fuel_types = ['Gasoline', 'Diesel', 'Electric', 'Hybrid']
    
    data = {
        'brand': np.random.choice(brands, n_samples),
        'mileage': np.random.randint(0, 300000, n_samples),
        'year': np.random.randint(2010, 2025, n_samples),
        'fuel_type': np.random.choice(fuel_types, n_samples)
    }
    
    df = pd.DataFrame(data)
    brand_price = {'Toyota': 20000, 'VW': 25000, 'BMW': 35000, 'Ford': 18000, 'Tesla': 45000, 'Audi': 32000}
    fuel_multiplier = {'Gasoline': 1.0, 'Diesel': 0.95, 'Electric': 1.2, 'Hybrid': 1.1}
    
    df['price'] = df['brand'].map(brand_price) * df['fuel_type'].map(fuel_multiplier)
    df['price'] *= (0.95 ** (2025 - df['year']))
    df['price'] *= (0.98 ** (df['mileage'] / 10000))
    df['price'] += np.random.normal(0, 1000, n_samples)
    df['price'] = df['price'].clip(lower=1000).round(2)
    return df

if __name__ == "__main__":
    df = generate_car_data()
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Dataset generated at: {OUTPUT_PATH}")
