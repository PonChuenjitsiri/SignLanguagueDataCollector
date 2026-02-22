import serial
import pandas as pd
import numpy as np
import xgboost as xgb
from scipy.interpolate import interp1d
import os
from gtts import gTTS
import pygame
import io

# ======================================================
# 1. Configuration
# ======================================================
SERIAL_PORT = "COM3"
BAUD_RATE = 115200
MODEL_PATH = "gesture_model.json"
TARGET_FRAMES = 100  # ต้องตรงกับตอนเทรน

# ตารางแปลชื่อ Class เป็นภาษาไทย (เสียงพูด)
TRANSLATION_DICT = {
    "hello": "สวัสดี",
    "thank_you": "ขอบคุณ",
    "sorry": "ขอโทษ",
    "yes": "ใช่",
    "no_right": "ไม่ค่ะ",
    "no_left": "ไม่ครับ",
    "help": "ช่วยด้วย",
    "hurt": "เจ็บ",
    "hungry_right": "หิวค่ะ",
    "hungry_left": "หิวครับ",
    "i_am_full": "อิ่ม",
    "water": "น้ำ",
    "toilet": "ห้องน้ำ",
    "go": "ไป",
    "come_here": "มา",
    "wait": "รอ",
    "today": "วันนี้",
    "tomorrow": "พรุ่งนี้",
    "me": "ฉัน",
    "you": "คุณ",
    "home": "บ้าน",

}

# ======================================================
# 2. Initialize Audio (Thai Female Voice)
# ======================================================
pygame.mixer.init()

def speak_thai(text):
    """ฟังก์ชันสร้างเสียงพูดภาษาไทยและเล่นออกลำโพงโดยไม่สร้างไฟล์ลงเครื่อง"""
    try:
        tts = gTTS(text=text, lang='th')
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        pygame.mixer.music.load(fp)
        pygame.mixer.music.play()
    except Exception as e:
        print(f"Voice Error: {e}")

# ======================================================
# 3. Load Model & Prepare Labels
# ======================================================
try:
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    # สร้าง Map ลำดับ Class ให้ตรงกับ Dictionary (0: hello, 1: thank_you, ...)
    LABELS_MAP = {i: label for i, label in enumerate(TRANSLATION_DICT.keys())}
    print(f"--- Model Loaded: {MODEL_PATH} ---")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# ======================================================
# 4. Core Prediction Logic
# ======================================================
def resample_and_predict(data):
    data_np = np.array(data)
    current_len = data_np.shape[0]
    
    x_old = np.linspace(0, 1, current_len)
    x_new = np.linspace(0, 1, TARGET_FRAMES)
    
    resampled_data = []
    for col in range(22):
        f = interp1d(x_old, data_np[:, col], kind='linear', fill_value="extrapolate")
        resampled_data.append(f(x_new))
    
    resampled_np = np.array(resampled_data).T 
    
    input_vector = resampled_np.flatten().reshape(1, -1)
    
    idx = model.predict(input_vector)[0]
    probs = model.predict_proba(input_vector)[0]
    confidence = probs[idx]
    
    return LABELS_MAP[idx], confidence

# ======================================================
# 5. Main Serial Loop
# ======================================================
def main():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        ser.flushInput()
        print(f"--- Inference Server Ready on {SERIAL_PORT} ---")
        print("Waiting for gesture signal...")

        gesture_buffer = []
        is_collecting = False

        while True:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if not line: continue

            # ตรวจจับสัญญาณเริ่มจากถุงมือ
            if "START_SIGNAL" in line:
                print("\n[*] Detecting...", end="", flush=True)
                gesture_buffer = []
                is_collecting = True

            # ตรวจจับสัญญาณยกเลิกจากมือซ้าย
            elif "CANCEL_SIGNAL" in line:
                print(" -> [CANCELLED]")
                is_collecting = False
                gesture_buffer = []

            # รับข้อมูลแต่ละเฟรม
            elif is_collecting and (line.startswith("S ") or (line and line[0].isdigit())):
                parts = [x for x in line.split() if x not in ["S", "E"]]
                if len(parts) == 22:
                    gesture_buffer.append([float(x) for x in parts])
                    print(".", end="", flush=True)

            # เมื่อได้รับสัญญาณจบการทำท่า
            elif "SUCCESS_SIGNAL" in line:
                actual_frames = len(gesture_buffer)
                print(f" Done ({actual_frames} frames)")
                
                if actual_frames >= 10:
                    label_en, conf = resample_and_predict(gesture_buffer)
                    thai_text = TRANSLATION_DICT.get(label_en, "ไม่ทราบท่าทางค่ะ")
                    
                    print(f"\n" + "="*35)
                    print(f" RESULT  : {thai_text}")
                    print(f" CONF    : {conf*100:.2f}%")
                    print("="*35)
                    
                    # ถ้าความมั่นใจสูงพอ ให้พูดออกลำโพง
                    if conf > 0.45:
                        speak_thai(thai_text)
                    else:
                        print("[!] Confidence too low to speak.")
                else:
                    print("\n[!] Error: Gesture too short.")
                
                is_collecting = False
                print("\nReady for next gesture...")

    except KeyboardInterrupt:
        print("\nServer Exit...")
    except Exception as e:
        print(f"\nSerial/Main Error: {e}")

if __name__ == "__main__":
    main()