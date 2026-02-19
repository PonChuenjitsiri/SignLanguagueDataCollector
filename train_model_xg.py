import numpy as np
import pandas as pd
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from scipy.interpolate import interp1d
import joblib
import json

# ======================================================
# 1. Configuration
# ======================================================
DATA_DIR = "dataset" #
EXPECTED_FRAMES = 100 
MODEL_NAME = "gesture_model.json"
LABELS_FILE = "labels_map.json"

# ======================================================
# 2. Dynamic Labels Mapping (ดึงชื่อโฟลเดอร์อัตโนมัติ)
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

print("\n--- Detected Labels ---")
for k, v in LABELS_MAP.items():
    print(f"[{k}] : {v}")

# เซฟ Labels เก็บไว้ใช้ตอน Predict
with open(LABELS_FILE, "w", encoding="utf-8") as f:
    json.dump(LABELS_MAP, f, ensure_ascii=False, indent=4)
print(f"-> Saved labels to '{LABELS_FILE}'")
print("-----------------------\n")

# ======================================================
# 3. Resample Function
# ======================================================
def resample_gesture(data, target=100):
    data_np = np.array(data)
    non_zero_data = data_np[~np.all(data_np == 0, axis=1)]
    current_len = non_zero_data.shape[0]
    if current_len < 2: 
        return None
    old_x = np.linspace(0, current_len - 1, num=current_len)
    new_x = np.linspace(0, current_len - 1, num=target)
    f = interp1d(old_x, non_zero_data, axis=0, kind='linear', fill_value="extrapolate")
    return f(new_x)

# ======================================================
# 4. Load & Flatten Data 
# ======================================================
def load_dataset():
    X, y = [], []
    print(f"--- Loading raw data and Resampling to {EXPECTED_FRAMES} frames ---")
    for label_name in LABELS_MAP.values():
        path = os.path.join(DATA_DIR, label_name)
        files = [f for f in os.listdir(path) if f.endswith('.csv')]
        print(f"   {label_name}: {len(files)} files")
        
        for file in files:
            try:
                df = pd.read_csv(os.path.join(path, file))
                resampled_data = resample_gesture(df.values, target=EXPECTED_FRAMES)
                
                if resampled_data is not None:
                    X.append(resampled_data.flatten())
                    y.append(INV_LABELS_MAP[label_name])
                else:
                    print(f"      [SKIP] {file}: Data too short or empty after removing zeros.")
                    
            except Exception as e:
                print(f"      [ERROR] reading {file}: {e}")
                
    return np.array(X), np.array(y)

# ======================================================
# 5. Training Process
# ======================================================
X, y = load_dataset()

if len(X) == 0:
    print("\n[!] Error: ไม่พบข้อมูลสำหรับการเทรนเลยครับ")
    exit()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

print(f"\nFeature Count: {X.shape[1]}")
print(f"Training on {len(X_train)} samples...")

model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    objective='multi:softprob',
    num_class=len(LABELS_MAP),
    eval_metric='mlogloss'
)

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')

print(f"\nAccuracy per fold: {scores}")
print(f"Mean CV Accuracy: {scores.mean()*100:.2f}%")

model.fit(
    X_train, 
    y_train,
    eval_set=[(X_test, y_test)], 
    verbose=True                 
)

# Evaluation
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("\n" + "="*40)
print(f" Final Accuracy: {acc*100:.2f}%")
print("="*40)
print(classification_report(y_test, y_pred, target_names=list(LABELS_MAP.values())))

model.save_model(MODEL_NAME)
print(f"\n[DONE] Model saved as '{MODEL_NAME}'")