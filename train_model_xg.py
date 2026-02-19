import numpy as np
import pandas as pd
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ======================================================
# 1. Configuration
# ======================================================
DATA_DIR = "dataset"
EXPECTED_FRAMES = 50 
MODEL_NAME = "gesture_model.json"

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
# 2. Load & Flatten Data (The "Normal" Way)
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
                    X.append(df.values.flatten())
                    y.append(INV_LABELS_MAP[label_name])
            except Exception as e:
                print(f"      Error reading {file}: {e}")
                
    return np.array(X), np.array(y)

# ======================================================
# 3. Training Process
# ======================================================
X, y = load_dataset()

# แบ่งข้อมูล 85/15
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

print(f"\nFeature Count: {X.shape[1]}") # ควรได้ 1100
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

print(f"Accuracy per fold: {scores}")
print(f"Mean Accuracy: {scores.mean()*100:.2f}%")

model.fit(
    X_train, 
    y_train,
    eval_set=[(X_test, y_test)], # <--- ส่งข้อสอบให้โมเดลดูด้วย
    verbose=True                 # (Optional) แสดงผลคะแนนแต่ละรอบออกมาดู
)

# Evaluation
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("\n" + "="*40)
print(f" Final Accuracy: {acc*100:.2f}%")
print("="*40)
print(classification_report(y_test, y_pred, target_names=LABELS_MAP.values()))

model.save_model(MODEL_NAME)
print(f"\n[DONE] Model saved as '{MODEL_NAME}'")