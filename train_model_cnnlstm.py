import numpy as np
import pandas as pd
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ======================================================
# 1. Configuration
# ======================================================
DATA_DIR = "dataset_cf"
EXPECTED_FRAMES = 70 
NUM_FEATURES = 22 
MODEL_NAME = "gesture_model_cnnlstm.pth" # PyTorch ใช้นามสกุล .pth
LABELS_FILE = "labels_map.json"

# ======================================================
# 2. Dynamic Labels Mapping
# ======================================================
folder_names = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
folder_names.sort()

LABELS_MAP = {i: name for i, name in enumerate(folder_names)}
INV_LABELS_MAP = {v: k for k, v in LABELS_MAP.items()}

print("--- Detected Labels ---")
for k, v in LABELS_MAP.items():
    print(f"[{k}] : {v}")

with open(LABELS_FILE, "w", encoding="utf-8") as f:
    json.dump(LABELS_MAP, f, ensure_ascii=False, indent=4)

# ======================================================
# 3. Resample & Load Data (Zero-Starting)
# ======================================================
def resample_gesture(data, target=70):
    data_np = np.array(data)
    non_zero_data = data_np[~np.all(data_np == 0, axis=1)]
    current_len = non_zero_data.shape[0]
    if current_len < 2: return None
    old_x = np.linspace(0, current_len - 1, num=current_len)
    new_x = np.linspace(0, current_len - 1, num=target)
    f = interp1d(old_x, non_zero_data, axis=0, kind='linear', fill_value="extrapolate")
    return f(new_x)

X, y = [], []
for label_name in LABELS_MAP.values():
    path = os.path.join(DATA_DIR, label_name)
    files = [f for f in os.listdir(path) if f.endswith('.csv')]
    for file in files:
        try:
            df = pd.read_csv(os.path.join(path, file))
            resampled_data = resample_gesture(df.values, target=EXPECTED_FRAMES)
            if resampled_data is not None:
                normalized_data = resampled_data - resampled_data[0] # Zero-Starting
                X.append(normalized_data)
                y.append(INV_LABELS_MAP[label_name])
        except:
            pass

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int64)

# แบ่งข้อมูล
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# แปลงเป็น PyTorch Tensors
train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ======================================================
# 4. Build CNN-LSTM Model (PyTorch)
# ======================================================
class CNNLSTM(nn.Module):
    def __init__(self, num_classes):
        super(CNNLSTM, self).__init__()
        # Conv1D ใน PyTorch รับ Input แบบ (Batch, Channels, Length)
        self.conv1 = nn.Conv1d(in_channels=22, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
        # LSTM
        self.lstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=2, batch_first=True, dropout=0.3)
        
        # Fully Connected
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # x shape: (Batch, 70 frames, 22 features) 
        # สลับแกนให้เข้ากับ Conv1d -> (Batch, 22, 70)
        x = x.permute(0, 2, 1) 
        
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        
        # สลับแกนกลับให้เข้ากับ LSTM -> (Batch, seq_len, features)
        x = x.permute(0, 2, 1)
        
        lstm_out, (hn, cn) = self.lstm(x)
        # เอาเฉพาะ output ของ frame สุดท้ายมาทายคลาส
        out = self.fc(self.dropout(lstm_out[:, -1, :]))
        return out

# ======================================================
# 5. Training Process
# ======================================================
model = CNNLSTM(num_classes=len(LABELS_MAP))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 100
best_acc = 0.0

print(f"\n--- เริ่มเทรนโมเดล PyTorch ({len(X_train)} samples) ---")
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    val_acc = correct / total
    
    if (epoch+1) % 5 == 0:
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Val Accuracy: {val_acc*100:.2f}%")
        
    # Save best model
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), MODEL_NAME)

# ======================================================
# 6. Evaluation
# ======================================================
print("\n" + "="*40)
print(f" Final Best Accuracy: {best_acc*100:.2f}%")
print("="*40)

# โหลดโมเดลตัวที่ดีที่สุดมาเทสต์
model.load_state_dict(torch.load(MODEL_NAME))
model.eval()
y_pred_list = []
with torch.no_grad():
    for inputs, _ in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        y_pred_list.extend(predicted.numpy())

print(classification_report(y_test, y_pred_list, target_names=list(LABELS_MAP.values())))
print(f"[DONE] Model saved as '{MODEL_NAME}'")