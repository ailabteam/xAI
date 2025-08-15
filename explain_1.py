# explain.py

import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import shap
import matplotlib.pyplot as plt
from tqdm import tqdm

from train import MLP

# --- Configuration ---
PROCESSED_DATA_PATH = "./processed_data/"
MODEL_PATH = "./models/nids_mlp_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Hyperparameters ---
INPUT_DIM = 70
HIDDEN_DIM1 = 128
HIDDEN_DIM2 = 64
OUTPUT_DIM = 1
NUM_SAMPLES_FOR_SHAP = 100
NUM_BACKGROUND_SAMPLES = 100

def load_model_and_data():
    print("Loading model and data...")
    X_test = pd.read_parquet(os.path.join(PROCESSED_DATA_PATH, 'X_test.parquet'))
    model = MLP(input_dim=INPUT_DIM, hidden_dim1=HIDDEN_DIM1, hidden_dim2=HIDDEN_DIM2, output_dim=OUTPUT_DIM)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.to(DEVICE)
    model.eval()
    print("Model and data loaded successfully.")
    return model, X_test

def create_predict_function(model):
    def predict_fn(x):
        x_tensor = torch.tensor(x, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            preds = model(x_tensor)
        return preds.cpu().numpy().flatten()
    return predict_fn

def generate_shap_explanations(model, X_test):
    print(f"\n--- Generating SHAP explanations using {NUM_SAMPLES_FOR_SHAP} test samples ---")
    
    background_data = shap.sample(X_test, NUM_BACKGROUND_SAMPLES, random_state=42)
    test_samples = X_test.sample(n=NUM_SAMPLES_FOR_SHAP, random_state=42)
    
    predict_function = create_predict_function(model)
    
    explainer = shap.KernelExplainer(predict_function, background_data)
    
    print("Calculating SHAP values with KernelExplainer... This might take a few minutes.")
    shap_values = explainer.shap_values(test_samples, nsamples="auto")

    # 1. & 2. Global Plots
    print("Generating and saving global summary plots...")
    plt.figure()
    shap.summary_plot(shap_values, test_samples, plot_type="bar", show=False)
    plt.savefig('shap_summary_bar_dpi600.png', dpi=600, bbox_inches='tight')
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, test_samples, show=False)
    plt.savefig('shap_summary_violin_dpi600.png', dpi=600, bbox_inches='tight')
    plt.close()
    print("Global plots saved.")

    # 3. Local Explanation: Force Plot
    print("\nGenerating and saving local force plot for an ATTACK sample...")
    
    preds = predict_function(test_samples.values)
    attack_index = np.where(preds > 0.5)[0]
    
    if len(attack_index) > 0:
        idx = attack_index[0]
        
        expected_value = explainer.expected_value
        if isinstance(expected_value, list):
             expected_value = expected_value[0]
        
        single_shap_value = shap_values[idx, :]
        
        plt.figure(figsize=(20, 5))
        shap.force_plot(expected_value, 
                       single_shap_value, 
                       test_samples.iloc[idx, :], 
                       matplotlib=True, 
                       show=False,
                       text_rotation=30)
                       
        plt.savefig('shap_force_plot_attack_dpi600_rotated.png', dpi=600, bbox_inches='tight')
        plt.close()
        print("Saved shap_force_plot_attack_dpi600_rotated.png")
    else:
        print("Could not find an ATTACK sample in the selected subset to generate a force plot.")

if __name__ == '__main__':
    model, X_test = load_model_and_data()
    generate_shap_explanations(model, X_test)
