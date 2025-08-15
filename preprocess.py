# preprocess.py

import pandas as pd
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# --- Configuration ---
DATASET_PATH = "./datasets/cicids2017/"
PROCESSED_DATA_PATH = "./processed_data/"
RANDOM_STATE = 42

def load_data(path):
    """Tải và gộp tất cả các file CSV từ một thư mục."""
    if not os.path.exists(path):
        print(f"Error: Directory not found at {path}")
        return None
        
    csv_files = glob.glob(os.path.join(path, "*.csv"))
    if not csv_files:
        print(f"Error: No CSV files found in {path}")
        return None

    print(f"Found {len(csv_files)} CSV files. Loading...")
    df = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)
    
    # Làm sạch tên cột (loại bỏ khoảng trắng thừa)
    df.columns = df.columns.str.strip()
    
    print("Data loading complete.")
    print(f"Initial shape: {df.shape}")
    return df

def clean_data(df):
    """Làm sạch DataFrame: xử lý infinity, NaN và các cột vô dụng."""
    print("\n--- Cleaning Data ---")
    
    # 1. Loại bỏ các giá trị vô cực (infinity)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # 2. Loại bỏ các hàng có giá trị thiếu (NaN)
    initial_rows = df.shape[0]
    df.dropna(inplace=True)
    print(f"Removed {initial_rows - df.shape[0]} rows with NaN values.")
    
    # 3. Loại bỏ các cột chỉ có một giá trị duy nhất (không có thông tin)
    for col in df.columns:
        if len(df[col].unique()) == 1:
            df.drop(col, inplace=True, axis=1)
            print(f"Removed single-value column: {col}")
            
    print(f"Shape after cleaning: {df.shape}")
    return df

def preprocess_data(df):
    """Tiền xử lý dữ liệu cho mô hình học máy."""
    print("\n--- Preprocessing Data ---")
    
    # 1. Tạo nhãn nhị phân (0 = BENIGN, 1 = ATTACK)
    df['Label'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
    print("Label distribution after binarization:")
    print(df['Label'].value_counts())
    
    # 2. Tách features (X) và nhãn (y)
    X = df.drop('Label', axis=1)
    y = df['Label']
    
    # 3. Chia dữ liệu thành tập train và test
    print("\nSplitting data into training and testing sets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    
    # 4. Chuẩn hóa features
    print("Scaling features using StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Chuyển đổi lại thành DataFrame để giữ tên cột (quan trọng cho SHAP sau này)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def save_processed_data(X_train, X_test, y_train, y_test, path):
    """Lưu các DataFrame đã xử lý vào file."""
    print(f"\n--- Saving Processed Data to {path} ---")
    os.makedirs(path, exist_ok=True)
    
    # Lưu dưới dạng Parquet để hiệu quả hơn CSV
    X_train.to_parquet(os.path.join(path, 'X_train.parquet'))
    X_test.to_parquet(os.path.join(path, 'X_test.parquet'))
    y_train.to_frame().to_parquet(os.path.join(path, 'y_train.parquet'))
    y_test.to_frame().to_parquet(os.path.join(path, 'y_test.parquet'))
    
    print("Processed data saved successfully.")

if __name__ == '__main__':
    # Chạy toàn bộ quy trình
    
    # Bước 1: Tải dữ liệu
    raw_df = load_data(DATASET_PATH)
    
    if raw_df is not None:
        # Bước 2: Làm sạch dữ liệu
        cleaned_df = clean_data(raw_df)
        
        # Bước 3: Tiền xử lý
        X_train, X_test, y_train, y_test = preprocess_data(cleaned_df)
        
        # Bước 4: Lưu kết quả
        save_processed_data(X_train, X_test, y_train, y_test, PROCESSED_DATA_PATH)
