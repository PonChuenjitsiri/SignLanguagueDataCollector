import numpy as np
import pandas as pd
import os
# import xgboost as xgb  <-- ไม่ใช้แล้ว
from sklearn.ensemble import RandomForestClassifier # <-- เพิ่มตัวนี้
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ======================================================
# 1. Configuration
# ======================================================
DATA_DIR = "dataset"
EXPECTED_FRAMES = 50 
MODEL_NAME = "gesture_model_rf.pkl" # <-- เปลี่ยนนามสกุลเป็น .pkl หรือ .joblib

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
# 2. Load & Flatten Data
# ======================================================
def load_dataset():
    X, y = [], []
    print(f"--- Loading raw data ({EXPECTED_FRAMES} frames x 22 sensors) ---")
    for label_name in LABELS_MAP.values():
        path = os.path.join(DATA_DIR, label_name)
        if not os.path.exists(path): continue
        files = [f for f in os.listdir(path) if f.endswith('.csv')]
        print(f"   {label_name}: {len(files)} files")
        
        for file in files:
            try:
                df = pd.read_csv(os.path.join(path, file))
                if len(df) == EXPECTED_FRAMES:
                    # คลี่ข้อมูล 50x22 เป็นเวกเตอร์แถวเดียว 1,100 ฟีเจอร์
                    X.append(extract_advanced_features(df))
                    y.append(INV_LABELS_MAP[label_name])
            except Exception as e:
                print(f"      Error reading {file}: {e}")
                
    return np.array(X), np.array(y)

def extract_advanced_features(df):
    raw_data = df.values # shape (50, 22)
    
    # 1. Raw Data Flatten (Original) -> 1100 features
    feat_raw = raw_data.flatten()
    
    # 2. Velocity (ความเปลี่ยนแปลงเทียบเฟรมต่อเฟรม) -> 22 sensors
    # ใช้ np.diff แล้วหาค่าเฉลี่ยความเร็ว หรือจะ flatten ความเร็วทั้งหมดก็ได้
    # แต่เพื่อประหยัด feature แนะนำให้เอา 'Mean Velocity' และ 'Max Velocity' ของแต่ละเซนเซอร์
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
# 3. Training Process (Random Forest)
# ======================================================
X, y = load_dataset()

# แบ่งข้อมูล 85/15
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

print(f"\nFeature Count: {X.shape[1]}")
print(f"Training on {len(X_train)} samples...")

# --- ส่วนที่เปลี่ยนแปลง: สร้างโมเดล Random Forest ---
model = RandomForestClassifier(
    n_estimators=100,      # จำนวนต้นไม้
    criterion='gini',      # เกณฑ์การวัด (gini หรือ entropy)
    max_depth=None,        # ปล่อยให้ลึกสุดได้ (หรือจะกำหนดเช่น 10, 20 ก็ได้เพื่อกัน Overfit)
    random_state=42,
    n_jobs=-1              # ใช้ทุก CPU core เพื่อความเร็ว
)

# Cross Validation
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')

print(f"Accuracy per fold: {scores}")
print(f"Mean Accuracy: {scores.mean()*100:.2f}%")

# --- ส่วนที่เปลี่ยนแปลง: การ Fit ---
# Random Forest ไม่ใช้ eval_set ในคำสั่ง fit เหมือน XGBoost
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("\n" + "="*40)
print(f" Final Accuracy: {acc*100:.2f}%")
print("="*40)
print(classification_report(y_test, y_pred, target_names=LABELS_MAP.values()))

# --- ส่วนที่เปลี่ยนแปลง: การ Save ---
# ใช้ joblib แทน model.save_model
joblib.dump(model, MODEL_NAME)
print(f"\n[DONE] Model saved as '{MODEL_NAME}'")