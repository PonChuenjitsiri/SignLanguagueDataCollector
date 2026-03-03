import numpy as np
import pandas as pd
import os
import json
import random
import xgboost as xgb
from scipy.interpolate import interp1d

# ======================================================
# 1. Configuration (ต้องตรงกับตอนเทรน)
# ======================================================
DATA_DIR = "dataset_cf"
EXPECTED_FRAMES = 70 
MODEL_NAME = "gesture_model.json"
LABELS_FILE = "labels_map.json"

# ======================================================
# 2. Resample Function (ใช้ฟังก์ชันเดิมเพื่อให้ข้อมูลหน้าตาเหมือนตอนเทรน)
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
# 3. Load Labels & Prompt User
# ======================================================
if not os.path.exists(LABELS_FILE):
    print(f"[!] ไม่พบไฟล์ {LABELS_FILE} กรุณารันโค้ดเทรนก่อนครับ")
    exit()

with open(LABELS_FILE, "r", encoding="utf-8") as f:
    # JSON จะอ่าน Key ที่เป็นตัวเลขเป็น String เสมอ (เช่น "0", "1")
    LABELS_MAP = json.load(f)

print("\n=== Available Labels ===")
for k, v in LABELS_MAP.items():
    print(f" [{k}] : {v}")
print("========================")

# รับค่าจากผู้ใช้
selected_idx = input("\nพิมพ์ตัวเลข Label ที่ต้องการสุ่มทดสอบ: ").strip()

if selected_idx not in LABELS_MAP:
    print("[!] ตัวเลขไม่ถูกต้อง หรือไม่มีในระบบครับ")
    exit()

target_class_name = LABELS_MAP[selected_idx]
folder_path = os.path.join(DATA_DIR, target_class_name)

if not os.path.exists(folder_path):
    print(f"[!] ไม่พบโฟลเดอร์ข้อมูล {folder_path}")
    exit()

# ======================================================
# 4. Randomly Select a File
# ======================================================
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

if not csv_files:
    print(f"[!] ไม่พบไฟล์ .csv ในโฟลเดอร์ {folder_path}")
    exit()

random_file = random.choice(csv_files)
file_path = os.path.join(folder_path, random_file)
print(f"\n-> สุ่มได้ไฟล์: {random_file} (คลาสที่แท้จริง: {target_class_name})")

# ======================================================
# 5. Load & Preprocess Data
# ======================================================
try:
    df = pd.read_csv(file_path)
    resampled_data = resample_gesture(df.values, target=EXPECTED_FRAMES)
    
    if resampled_data is None:
        print("[!] ไฟล์นี้มีข้อมูลไม่พอหลังจากตัด 0 ออกครับ รบกวนสุ่มใหม่")
        exit()
        
    # Flatten ข้อมูลให้เป็น 1D Array และ reshape ให้เป็น 2D (1 sample, n features) ตามที่ XGBoost ต้องการ
    X_input = resampled_data.flatten().reshape(1, -1)
    
except Exception as e:
    print(f"[!] เกิดข้อผิดพลาดในการอ่านไฟล์: {e}")
    exit()

# ======================================================
# 6. Load Model & Predict
# ======================================================
if not os.path.exists(MODEL_NAME):
    print(f"[!] ไม่พบโมเดล {MODEL_NAME} กรุณาเทรนก่อนครับ")
    exit()

# โหลดโมเดล
model = xgb.XGBClassifier()
model.load_model(MODEL_NAME)

# ทำนายผล
predicted_idx = model.predict(X_input)[0]
# ดึงค่าความมั่นใจ (Probability)
probabilities = model.predict_proba(X_input)[0]
confidence = probabilities[predicted_idx] * 100

predicted_class_name = LABELS_MAP[str(predicted_idx)]

print("\n" + "="*40)
print("             RESULTS")
print("="*40)
print(f" Actual Class    : {target_class_name}")
print(f" Predicted Class : {predicted_class_name}")
print(f" Confidence      : {confidence:.2f}%")

if target_class_name == predicted_class_name:
    print("\n✅ ทายถูก!")
else:
    print("\n❌ ทายผิด!")
print("="*40 + "\n")