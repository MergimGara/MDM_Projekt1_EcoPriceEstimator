import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, StringTensorType

# Resolve paths
MODEL_DIR = Path(__file__).resolve().parent
BASE_DIR = MODEL_DIR.parent
DATA_PATH = BASE_DIR / "data" / "dataset.csv"

def train_and_export():
    if not DATA_PATH.exists():
        print(f"Error: Data not found at {DATA_PATH}. Run generate_dataset.py first.")
        return

    df = pd.read_csv(DATA_PATH)
    X = df.drop('price', axis=1)
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    numeric_features = ['mileage', 'year']
    categorical_features = ['brand', 'fuel_type']
    
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(n_estimators=100, random_state=42))
    ])
    
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"Model R2 Score: {score:.4f}")
    
    initial_type = [
        ('brand', StringTensorType([None, 1])),
        ('mileage', FloatTensorType([None, 1])),
        ('year', FloatTensorType([None, 1])),
        ('fuel_type', StringTensorType([None, 1]))
    ]
    
    onx = convert_sklearn(model, initial_types=initial_type, target_opset=12)
    
    with open(MODEL_DIR / "model.onnx", "wb") as f:
        f.write(onx.SerializeToString())
    
    metadata = {
        'brands': sorted(df['brand'].unique().tolist()),
        'fuel_types': sorted(df['fuel_type'].unique().tolist()),
        'r2_score': float(score)
    }
    
    with open(MODEL_DIR / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Model and metadata saved in: {MODEL_DIR}")

if __name__ == "__main__":
    train_and_export()
