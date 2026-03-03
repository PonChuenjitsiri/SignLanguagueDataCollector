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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# ======================================================
# 1. Configuration
# ======================================================
DATA_DIR = "dataset_cf"
EXPECTED_FRAMES = 70 
NUM_FEATURES = 22 
PYTORCH_MODEL_NAME = "gesture_model_best_cnnlstm.pth"
XGB_MODEL_NAME = "gesture_model_best_xgb.json"
LABELS_FILE = "labels_map.json"

# ======================================================
# 2. Dynamic Labels Mapping
# ======================================================
if not os.path.exists(DATA_DIR):
    print(f"[!] ไม่พบโฟลเดอร์ {DATA_DIR} กรุณาสร้างและใส่ข้อมูลก่อนครับ")
    exit()

folder_names = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
folder_names.sort()

if len(folder_names) == 0:
    print(f"[!] ไม่พบโฟลเดอร์ย่อยใน {DATA_DIR} เลยครับ")
    exit()

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
print(f"\n--- Loading raw data and Resampling to {EXPECTED_FRAMES} frames ---")
for label_name in LABELS_MAP.values():
    path = os.path.join(DATA_DIR, label_name)
    files = [f for f in os.listdir(path) if f.endswith('.csv')]
    print(f"   {label_name}: {len(files)} files")
    for file in files:
        try:
            df = pd.read_csv(os.path.join(path, file))
            resampled_data = resample_gesture(df.values, target=EXPECTED_FRAMES)
            if resampled_data is not None:
                normalized_data = resampled_data - resampled_data[0] # Zero-Starting
                X.append(normalized_data)
                y.append(INV_LABELS_MAP[label_name])
        except Exception as e:
            pass

X_3d = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int64)

if len(X_3d) == 0:
    print("\n[!] Error: ไม่พบข้อมูลสำหรับการเทรนเลยครับ")
    exit()

# แบ่งข้อมูลครั้งเดียว เพื่อให้ทั้ง CNN-LSTM และ XGBoost ใช้ Train/Test ชุดเดียวกันเป๊ะๆ
X_train_3d, X_test_3d, y_train, y_test = train_test_split(X_3d, y, test_size=0.3, random_state=42, stratify=y)

# Flatten ข้อมูลสำหรับ XGBoost (N, 70, 22) -> (N, 1540)
X_train_2d = X_train_3d.reshape(X_train_3d.shape[0], -1)
X_test_2d = X_test_3d.reshape(X_test_3d.shape[0], -1)

print(f"\n--- Data Shapes ---")
print(f"CNN-LSTM Shape: Train={X_train_3d.shape}, Test={X_test_3d.shape}")
print(f"XGBoost Shape:  Train={X_train_2d.shape}, Test={X_test_2d.shape}")

# เตรียม DataLoader สำหรับ PyTorch
train_dataset = TensorDataset(torch.tensor(X_train_3d), torch.tensor(y_train))
test_dataset = TensorDataset(torch.tensor(X_test_3d), torch.tensor(y_test))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False) # ห้าม Shuffle เพื่อรักษาลำดับตอนเทสต์

# ======================================================
# 4. Build CNN-LSTM Model (PyTorch)
# ======================================================
class CNNLSTM(nn.Module):
    def __init__(self, num_classes):
        super(CNNLSTM, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=22, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.lstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=2, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1) # (Batch, 22, 70)
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = x.permute(0, 2, 1) # (Batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        out = self.fc(self.dropout(lstm_out[:, -1, :]))
        return out

# ======================================================
# 5. Training CNN-LSTM
# ======================================================
cnn_lstm_model = CNNLSTM(num_classes=len(LABELS_MAP))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn_lstm_model.parameters(), lr=0.001)

epochs = 100
best_acc = 0.0

print(f"\n--- เริ่มเทรนโมเดล CNN-LSTM (PyTorch) ---")
for epoch in range(epochs):
    cnn_lstm_model.train()
    total_loss = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = cnn_lstm_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    # Validation
    cnn_lstm_model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = cnn_lstm_model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    val_acc = correct / total
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Val Accuracy: {val_acc*100:.2f}%")
        
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(cnn_lstm_model.state_dict(), PYTORCH_MODEL_NAME)

print(f"[DONE] CNN-LSTM Best Val Accuracy: {best_acc*100:.2f}%")

# โหลดโมเดล CNN-LSTM ตัวที่ดีที่สุดมาเตรียมไว้
cnn_lstm_model.load_state_dict(torch.load(PYTORCH_MODEL_NAME))
cnn_lstm_model.eval()

# ======================================================
# 6. Training XGBoost
# ======================================================
print("\n--- เริ่มเทรนโมเดล XGBoost ---")
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    objective='multi:softprob',
    num_class=len(LABELS_MAP),
    eval_metric='mlogloss',
    random_state=42
)

xgb_model.fit(
    X_train_2d, 
    y_train,
    eval_set=[(X_test_2d, y_test)], 
    verbose=10 # โชว์ผลทุกๆ 10 ต้น
)

xgb_model.save_model(XGB_MODEL_NAME)
print(f"[DONE] XGBoost Model saved as '{XGB_MODEL_NAME}'")

# ======================================================
# 7. Extract Probabilities for Soft Voting Ensemble
# ======================================================
print("\n--- เริ่มกระบวนการทำ Ensemble (Soft Voting) ---")

# 7.1 ดึง Probability จาก CNN-LSTM
cnn_lstm_probs_list = []
with torch.no_grad():
    for inputs, _ in test_loader:
        outputs = cnn_lstm_model(inputs)
        # เนื่องจาก PyTorch คืนค่าเป็น Raw Logits จึงต้องใช้ Softmax เพื่อแปลงเป็น Probability (0-1)
        probs = torch.softmax(outputs, dim=1) 
        cnn_lstm_probs_list.extend(probs.numpy())
probs_cnn_lstm = np.array(cnn_lstm_probs_list)

# 7.2 ดึง Probability จาก XGBoost
probs_xgboost = xgb_model.predict_proba(X_test_2d)

# 7.3 นำ Probability มาเฉลี่ยกัน (น้ำหนัก 50:50)
ensemble_probs = (probs_cnn_lstm + probs_xgboost) / 2.0

# 7.4 หาคลาสที่คะแนนโหวตสูงสุด
preds_cnn_lstm = np.argmax(probs_cnn_lstm, axis=1)
preds_xgboost = np.argmax(probs_xgboost, axis=1)
preds_ensemble = np.argmax(ensemble_probs, axis=1)

# ======================================================
# 8. Evaluation & Comparison
# ======================================================
print("\n" + "="*50)
print("สรุปความแม่นยำ (Accuracy Comparison)")
print("="*50)
print(f"1. CNN-LSTM (PyTorch) Accuracy: {accuracy_score(y_test, preds_cnn_lstm)*100:.2f}%")
print(f"2. XGBoost Accuracy:            {accuracy_score(y_test, preds_xgboost)*100:.2f}%")
print(f"3. Ensemble (Soft Voting):      {accuracy_score(y_test, preds_ensemble)*100:.2f}%")
print("="*50)

print("\n--- Classification Report ของ Ensemble ---")
print(classification_report(y_test, preds_ensemble, target_names=list(LABELS_MAP.values())))

# ======================================================
# 9. Plot Confusion Matrix
# ======================================================
def plot_cm(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=list(LABELS_MAP.values()), 
                yticklabels=list(LABELS_MAP.values()))
    plt.title(title)
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# พล็อตเทียบกันเพื่อดูจุดบอดที่ถูกแก้
plot_cm(y_test, preds_cnn_lstm, "CNN-LSTM Confusion Matrix")
plot_cm(y_test, preds_xgboost, "XGBoost Confusion Matrix")
plot_cm(y_test, preds_ensemble, "Ensemble (Soft Voting) Confusion Matrix")