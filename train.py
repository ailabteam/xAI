# train.py

import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
PROCESSED_DATA_PATH = "./processed_data/"
MODEL_PATH = "./models/"
RANDOM_STATE = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Hyperparameters ---
INPUT_DIM = 70  # Sẽ được xác nhận lại khi tải dữ liệu
HIDDEN_DIM1 = 128
HIDDEN_DIM2 = 64
OUTPUT_DIM = 1
LEARNING_RATE = 0.001
BATCH_SIZE = 1024
EPOCHS = 10

def load_processed_data(path):
    """Tải dữ liệu đã được tiền xử lý."""
    print(f"Loading processed data from {path}...")
    X_train = pd.read_parquet(os.path.join(path, 'X_train.parquet'))
    X_test = pd.read_parquet(os.path.join(path, 'X_test.parquet'))
    y_train = pd.read_parquet(os.path.join(path, 'y_train.parquet'))['Label']
    y_test = pd.read_parquet(os.path.join(path, 'y_test.parquet'))['Label']
    return X_train, X_test, y_train, y_test

def create_dataloaders(X_train, y_train, X_test, y_test, batch_size):
    """Tạo PyTorch DataLoaders."""
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim1)
        self.layer2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.layer3 = nn.Linear(hidden_dim2, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.sigmoid(self.layer3(x))
        return x

def train_model(model, train_loader, epochs, lr):
    """Huấn luyện mô hình."""
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.to(DEVICE)
    print(f"\n--- Starting Training on {DEVICE} for {epochs} epochs ---")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (features, labels) in enumerate(train_loader):
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")
    
    print("Training finished.")

def evaluate_model(model, test_loader):
    """Đánh giá mô hình và in kết quả."""
    model.eval()
    all_preds = []
    all_labels = []
    
    print("\n--- Evaluating Model ---")
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            outputs = model(features)
            
            predicted = (outputs > 0.5).float()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()
    
    print("\n--- Model Evaluation Results ---")
    print(f"Accuracy: {accuracy_score(all_labels, all_preds):.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['BENIGN', 'ATTACK']))
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['BENIGN', 'ATTACK'], yticklabels=['BENIGN', 'ATTACK'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    
    output_filename = 'confusion_matrix_dpi600.png'
    plt.savefig(output_filename, dpi=600, bbox_inches='tight')
    
    print(f"\nConfusion Matrix saved to {output_filename}")

def save_model(model, path):
    """Lưu mô hình đã huấn luyện."""
    os.makedirs(path, exist_ok=True)
    model_save_path = os.path.join(path, 'nids_mlp_model.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_processed_data(PROCESSED_DATA_PATH)
    
    INPUT_DIM = X_train.shape[1]
    print(f"Input dimension set to: {INPUT_DIM}")
    
    train_loader, test_loader = create_dataloaders(X_train, y_train, X_test, y_test, BATCH_SIZE)
    
    model = MLP(input_dim=INPUT_DIM, hidden_dim1=HIDDEN_DIM1, hidden_dim2=HIDDEN_DIM2, output_dim=OUTPUT_DIM)
    
    train_model(model, train_loader, epochs=EPOCHS, lr=LEARNING_RATE)
    
    evaluate_model(model, test_loader)
    
    save_model(model, MODEL_PATH)
