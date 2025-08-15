import torch
import pandas as pd
import sklearn
import shap

print(f"PyTorch version: {torch.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"Scikit-learn version: {sklearn.__version__}")
print(f"SHAP version: {shap.__version__}")

# Kiểm tra xem PyTorch có nhận GPU không (nếu bạn cài bản GPU)
if torch.cuda.is_available():
    print(f"CUDA is available. Device: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. Using CPU.")
