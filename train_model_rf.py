import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import joblib
from scipy.interpolate import interp1d

# ======================================================
# 1. Configuration
# ======================================================
DATA_DIR = "dataset"
EXPECTED_FRAMES = 50 
MODEL_NAME = "gesture_model_rf.pkl"

LABELS_MAP = {
    0: "hello",
    1: "thank_you",
    2: "sorry",
    3: "yes",
    4: "no_right",
    5: "no_left",
    6: "help",
    7: "pain",
    8: "hungry_right",
    9: "hungry_left",
    10: "i'm_full",
    11: "water",
    12: "toilet",
    13: "go_right",
    14: "go_left",
}
INV_LABELS_MAP = {v: k for k, v in LABELS_MAP.items()}

# ======================================================
# 2. Resample Function
# ======================================================
def resample_gesture(data, target=50):
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
# 3. Feature Extraction
# ======================================================
def extract_advanced_features(raw_data):
    # ปรับให้รับค่าเป็น numpy array (เพราะได้มาจาก resample_gesture)
    # raw_data shape: (50, 22)
    
    # 1. Raw Data Flatten -> 1100 features
    feat_raw = raw_data.flatten()
    
    # 2. Velocity (ความเปลี่ยนแปลงเทียบเฟรมต่อเฟรม) -> 22 sensors
    velocity = np.diff(raw_data, axis=0) # shape (49, 22)
    feat_vel_mean = np.mean(velocity, axis=0) # เฉลี่ยความเร็วแต่ละนิ้ว
    feat_vel_std = np.std(velocity, axis=0)   # ความสม่ำเสมอของความเร็ว
    
    # 3. Statistical (ภาพรวมพฤติกรรม) -> 22 sensors each
    feat_mean = np.mean(raw_data, axis=0)
    feat_std = np.std(raw_data, axis=0)   # บอกว่านิ้วไหนขยับเยอะสุด
    feat_range = np.ptp(raw_data, axis=0) # Peak-to-Peak (Max - Min)
    
    # รวม Feature ทั้งหมดเข้าด้วยกัน
    # Raw(1100) + VelMean(22) + VelStd(22) + Mean(22) + Std(22) + Range(22)
    # Total = 1100 + 110 = 1210 Features
    features = np.concatenate([
        feat_raw, 
        feat_vel_mean, 
        feat_vel_std, 
        feat_mean, 
        feat_std, 
        feat_range
    ])
    
    return features

# ======================================================
# 4. Load & Process Data
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
                
                # นำข้อมูลดิบมารีแซมเปิลให้ได้ 50 Frames ก่อน
                resampled_data = resample_gesture(df.values, target=EXPECTED_FRAMES)
                
                if resampled_data is not None:
                    # นำข้อมูลที่รีแซมเปิลแล้ว ไปสกัดฟีเจอร์ 1,210 ตัว
                    features = extract_advanced_features(resampled_data)
                    X.append(features)
                    y.append(INV_LABELS_MAP[label_name])
                else:
                    print(f"      [SKIP] {file}: Data too short or empty after removing zeros.")
                    
            except Exception as e:
                print(f"      [ERROR] reading {file}: {e}")
                
    return np.array(X), np.array(y)

# ======================================================
# 5. Training Process (Random Forest)
# ======================================================
X, y = load_dataset()

if len(X) == 0:
    print("\n[!] Error: ไม่พบข้อมูลสำหรับการเทรนเลยครับ")
    exit()

# แบ่งข้อมูล 85/15
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

print(f"\nFeature Count: {X.shape[1]} (Expected 1210)")
print(f"Training on {len(X_train)} samples...")

model = RandomForestClassifier(
    n_estimators=100,      # จำนวนต้นไม้
    criterion='gini',      # เกณฑ์การวัด
    max_depth=None,        # ปล่อยให้ลึกสุดได้
    random_state=42,
    n_jobs=-1              # ใช้ทุก CPU core เพื่อความเร็ว
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
print(classification_report(y_test, y_pred, target_names=LABELS_MAP.values()))

# Save model
joblib.dump(model, MODEL_NAME)
print(f"\n[DONE] Model saved as '{MODEL_NAME}'")