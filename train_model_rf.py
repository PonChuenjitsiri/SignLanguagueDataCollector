import numpy as np
import pandas as pd
import os
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import joblib
from scipy.interpolate import interp1d

# ======================================================
# 1. Configuration
# ======================================================
DATA_DIR = "dataset_cf" # เปลี่ยนให้ตรงกับชื่อโฟลเดอร์ที่ใช้
EXPECTED_FRAMES = 70 
MODEL_NAME = "gesture_model_rf.pkl"
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
def resample_gesture(data, target=70):
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
# 4. Feature Extraction
# ======================================================
def extract_advanced_features(raw_data):
    # ไม่เอา raw_data.flatten() แล้ว เพื่อกันโมเดลจำ
    
    velocity = np.diff(raw_data, axis=0) 
    feat_vel_mean = np.mean(velocity, axis=0) 
    feat_vel_std = np.std(velocity, axis=0)   
    
    feat_mean = np.mean(raw_data, axis=0)
    feat_std = np.std(raw_data, axis=0)   
    feat_range = np.ptp(raw_data, axis=0) 
    
    # รวม Feature เหลือแค่ 22 * 5 = 110 Features (เบาลงเยอะและฉลาดขึ้น)
    features = np.concatenate([
        feat_vel_mean, 
        feat_vel_std, 
        feat_mean, 
        feat_std, 
        feat_range
    ])
    
    return features

# ======================================================
# 5. Load & Process Data
# ======================================================
def load_dataset():
    X, y = [], []
    print(f"--- Loading raw data and Resampling to {EXPECTED_FRAMES} frames ---")
    for label_name in LABELS_MAP.values():
        path = os.path.join(DATA_DIR, label_name)
        if not os.path.exists(path): continue
        files = [f for f in os.listdir(path) if f.endswith('.csv')]
        print(f"   {label_name}: {len(files)} files")
        
        for file in files:
            try:
                df = pd.read_csv(os.path.join(path, file))
                
                # นำข้อมูลดิบมารีแซมเปิลให้ได้ 70 Frames ก่อน
                resampled_data = resample_gesture(df.values, target=EXPECTED_FRAMES)
                
                if resampled_data is not None:
                    # นำข้อมูลที่รีแซมเปิลแล้ว ไปสกัดฟีเจอร์ 1,650 ตัว
                    features = extract_advanced_features(resampled_data)
                    X.append(features)
                    y.append(INV_LABELS_MAP[label_name])
                else:
                    print(f"      [SKIP] {file}: Data too short or empty after removing zeros.")
                    
            except Exception as e:
                print(f"      [ERROR] reading {file}: {e}")
                
    return np.array(X), np.array(y)

# ======================================================
# 6. Training Process (Random Forest)
# ======================================================
X, y = load_dataset()

if len(X) == 0:
    print("\n[!] Error: ไม่พบข้อมูลสำหรับการเทรนเลยครับ")
    exit()

# แบ่งข้อมูล 85/15 ให้เหมือน xgboost
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

print(f"\nFeature Count: {X.shape[1]} (Expected 1650 based on 70 frames)")
print(f"Training on {len(X_train)} samples...")

model = RandomForestClassifier(
    n_estimators=200,      # เพิ่มต้นไม้ให้ช่วยกันโหวตเยอะขึ้น
    criterion='gini',      
    max_depth=15,          # <-- เพิ่มความลึกให้พอแยก 22 คลาสได้ (จาก 3 เป็น 15)
    min_samples_split=5,   
    min_samples_leaf=2,    
    max_features='sqrt',   
    random_state=42,
    n_jobs=-1              
)

# Cross Validation
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')

print(f"\nAccuracy per fold: {scores}")
print(f"Mean CV Accuracy: {scores.mean()*100:.2f}%")

# Fit model
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("\n" + "="*40)
print(f" Final Test Accuracy: {acc*100:.2f}%")
print("="*40)
print(classification_report(y_test, y_pred, target_names=list(LABELS_MAP.values())))

# Save model
joblib.dump(model, MODEL_NAME)
print(f"\n[DONE] Model saved as '{MODEL_NAME}'")