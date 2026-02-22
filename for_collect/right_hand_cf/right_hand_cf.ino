#include <Arduino.h>
#include <Wire.h>
#include <MPU9250_asukiaaa.h>
#include <vector>
#include <Preferences.h>

HardwareSerial HC12(1);
#define HC12_RX 20
#define HC12_TX 21

const int PIN_LED = 10;
const int PIN_BUTTON = 5;

// --- Thresholds ---
const float T_ACCEL = 0.25; // กลับมาใช้ค่าคงที่
const float T_GYRO = 20.0;  // กลับมาใช้ค่าคงที่
int t_flex = 300;           // ค่าตั้งต้น (จะถูกปรับอัตโนมัติหลัง Calibrate)

int flexMin[5] = {0, 0, 0, 0, 0};      
int flexMax[5] = {4095, 4095, 4095, 4095, 4095}; 
bool isCalibrated = false;

Preferences preferences;

struct GloveData {
    uint16_t flex[5];
    int16_t accel[3];
    int16_t gyro[3];
};

MPU9250_asukiaaa mpu;

const uint8_t CMD_START = 0xA1;
const uint8_t CMD_STOP  = 0xA2;
const uint8_t CMD_DATA  = 0xD1;
const uint8_t CMD_END   = 0xD2;
const uint8_t SIG_CANCEL = 0xEE;

std::vector<GloveData> bufL, bufR;
GloveData lastDataR;
GloveData zeroData = { {0,0,0,0,0}, {0,0,0}, {0,0,0} };

enum State { IDLE, RECORDING, RECEIVING_LEFT };
State currentState = IDLE;

unsigned long btnPressStart = 0;
bool isBtnHeld = false;
bool actionTriggered = false;

// --- Memory Functions ---
void saveCalibrationToFlash() {
    preferences.begin("glove-cal", false);
    preferences.putBytes("fMin", flexMin, sizeof(flexMin));
    preferences.putBytes("fMax", flexMax, sizeof(flexMax));
    preferences.putInt("tFlex", t_flex);
    preferences.putBool("isCal", true);
    preferences.end();
    Serial.println(">> Flex Calibration Saved to Flash!");
}

void loadCalibrationFromFlash() {
    preferences.begin("glove-cal", true);
    isCalibrated = preferences.getBool("isCal", false);
    if (isCalibrated) {
        preferences.getBytes("fMin", flexMin, sizeof(flexMin));
        preferences.getBytes("fMax", flexMax, sizeof(flexMax));
        t_flex = preferences.getInt("tFlex", 200); // โหลด Threshold ขึ้นมาด้วย
        Serial.println(">> Calibration Loaded from Flash!");
        Serial.printf(">> Flex Threshold is set to: %d\n", t_flex);
    }
    preferences.end();
}

void blinkLED(int times, int duration) {
    for (int i = 0; i < times; i++) {
        digitalWrite(PIN_LED, HIGH); delay(duration);
        digitalWrite(PIN_LED, LOW); if (i < times - 1) delay(duration);
    }
}

void readMPU(GloveData &d) {
    mpu.accelUpdate();
    mpu.gyroUpdate();
    d.accel[0] = (int16_t)(mpu.accelX() * 100);
    d.accel[1] = (int16_t)(mpu.accelY() * 100);
    d.accel[2] = (int16_t)(mpu.accelZ() * 100);
    d.gyro[0] = (int16_t)(mpu.gyroX() * 100);
    d.gyro[1] = (int16_t)(mpu.gyroY() * 100);
    d.gyro[2] = (int16_t)(mpu.gyroZ() * 100);
}

void waitForUserAction() {
    while(digitalRead(PIN_BUTTON) == HIGH) delay(10);  
    delay(100); 
    while(digitalRead(PIN_BUTTON) == LOW) delay(10);   
    delay(100); 
    while(digitalRead(PIN_BUTTON) == HIGH) delay(10);  
    delay(100); 
}

void calibrateSensors() {
    Serial.println("\n=== CALIBRATION MODE (RIGHT HAND) ===");
    blinkLED(5, 100); digitalWrite(PIN_LED, HIGH);

    long sumOpen[5] = {0,0,0,0,0};
    long sumClose[5] = {0,0,0,0,0};

    // 1. Flex Calibration 5 รอบ
    for (int round = 1; round <= 5; round++) {
        Serial.printf(">> ROUND %d/5\n", round);
        
        Serial.println("   [ACTION] OPEN hand -> Press Button");
        waitForUserAction();
        digitalWrite(PIN_LED, LOW);
        for(int i=0; i<5; i++) sumOpen[i] += analogRead(4-i); 
        Serial.println("    Read Open Done.");

        Serial.println("   [ACTION] CLOSE hand -> Press Button");
        waitForUserAction();
        for(int i=0; i<5; i++) sumClose[i] += analogRead(4-i); 
        Serial.println("    Read Close Done.");

        blinkLED(2, 100);
        if(round < 5) digitalWrite(PIN_LED, HIGH);
    }

    Serial.println("\n=== CALIBRATION RESULTS ===");
    for(int i=0; i<5; i++) {
        flexMin[i] = sumOpen[i] / 5;
        flexMax[i] = sumClose[i] / 5;
        if (flexMin[i] == flexMax[i]) flexMax[i] += 1; // ป้องกัน หาร 0
        Serial.printf("  F%d Raw Avg Min: %d | Avg Max: %d\n", i, flexMin[i], flexMax[i]);
    }

    // 2. คำนวณ Threshold ใหม่ของ Flex (สเกลใหม่ 0-2000)
    // ใช้ 10% ของความกว้างสเกล (2000 * 10% = 200) เพื่อกรอง Noise ของนิ้ว
    t_flex = 200; 
    
    Serial.println("\n-----------------------------------------");
    Serial.println(">> Flex Mapped Range : 0 - 2000");
    Serial.printf(">> NEW Flex Threshold: %d (10%% of range)\n", t_flex);
    Serial.println("-----------------------------------------");
    
    isCalibrated = true;
    saveCalibrationToFlash(); // เซฟลงบอร์ด
    
    blinkLED(3, 200);
    while(digitalRead(PIN_BUTTON) == HIGH) delay(10);
    isBtnHeld = false; 
    actionTriggered = false;
}

String d2s(GloveData d) {
    char b[128];
    snprintf(b, sizeof(b), "%d %d %d %d %d %.2f %.2f %.2f %.2f %.2f %.2f",
             d.flex[0], d.flex[1], d.flex[2], d.flex[3], d.flex[4],
             d.accel[0]/100.0, d.accel[1]/100.0, d.accel[2]/100.0,
             d.gyro[0]/100.0, d.gyro[1]/100.0, d.gyro[2]/100.0);
    return String(b);
}

bool checkMovementR(GloveData current) {
    if (bufR.empty()) return true;
    for(int i=0; i<5; i++) if (abs((int)current.flex[i] - (int)lastDataR.flex[i]) > t_flex) return true;
    
    // กลับมาใช้ค่าคงที่เทียบเหมือนเดิม
    for(int k=0; k<3; k++) {
        if (abs(current.accel[k] - lastDataR.accel[k]) > (T_ACCEL * 100)) return true;
        if (abs(current.gyro[k] - lastDataR.gyro[k]) > (T_GYRO * 100)) return true;
    }
    return false;
}

void processAndPrint() {
    int maxFrames = max((int)bufL.size(), (int)bufR.size());
    if (maxFrames < 5) {
        Serial.println("DISCARD_SIGNAL"); 
        blinkLED(2, 50);
        return;
    }
    while(bufL.size() < maxFrames) bufL.push_back(bufL.size()>0 ? bufL.back() : zeroData);
    while(bufR.size() < maxFrames) bufR.push_back(bufR.size()>0 ? bufR.back() : zeroData);

    for (int i = 0; i < maxFrames; i++) {
        String line = (i==0) ? "S " : "";
        line += d2s(bufL[i]) + " " + d2s(bufR[i]);
        if (i==maxFrames-1) line += " E";
        Serial.println(line);
    }
    Serial.printf("SUCCESS_SIGNAL: %d FRAMES\n", maxFrames);
    blinkLED(3, 100);
}

void setup() {
    Serial.begin(115200);
    HC12.begin(115200, SERIAL_8N1, 20, 21);
    analogReadResolution(12);
    
    pinMode(PIN_BUTTON, INPUT); 
    pinMode(PIN_LED, OUTPUT);
    Wire.begin(6, 7);
    mpu.setWire(&Wire); mpu.beginAccel(); mpu.beginGyro();
    
    Serial.println("\n--- MASTER (RIGHT HAND) READY ---");
    loadCalibrationFromFlash();
}

void loop() {
    int btnState = digitalRead(PIN_BUTTON); 
    if (btnState == HIGH) { 
        if (!isBtnHeld) {
            isBtnHeld = true;
            btnPressStart = millis();
            actionTriggered = false;
        } else {
            if (millis() - btnPressStart > 3000 && !actionTriggered) {
                if (currentState == IDLE) { 
                    actionTriggered = true;
                    calibrateSensors();
                    isBtnHeld = false; 
                    btnPressStart = millis(); 
                    return; 
                }
            }
        }
    } else { 
        if (isBtnHeld) {
            if (!actionTriggered && (millis() - btnPressStart > 50)) { 
                if (currentState == IDLE) {
                    currentState = RECORDING;
                    bufL.clear(); bufR.clear();
                    HC12.write(CMD_START);
                    Serial.println("START_SIGNAL");
                    blinkLED(1, 100);
                } 
                else if (currentState == RECORDING) {
                    currentState = RECEIVING_LEFT;
                    HC12.write(CMD_STOP);
                    Serial.println("STOP_SIGNAL -> Waiting for Left Hand...");
                    blinkLED(1, 100);
                }
            }
            isBtnHeld = false;
        }
    }

    if (currentState == RECORDING) {
        static uint32_t last_scan = 0;
        if (millis() - last_scan >= 20) {
            last_scan = millis();
            if (mpu.accelUpdate() == 0 && mpu.gyroUpdate() == 0) {
                GloveData d;
                readMPU(d); 
                for(int i=0; i<5; i++) {
                    int raw = analogRead(4-i); 
                    if (isCalibrated) {
                        int clipped = constrain(raw, min(flexMin[i], flexMax[i]), max(flexMin[i], flexMax[i]));
                        // *** เปลี่ยน Range ขยายเป็น 0 - 2000 ***
                        d.flex[i] = map(clipped, flexMin[i], flexMax[i], 0, 2000);
                    } else {
                        d.flex[i] = raw; 
                    }
                }
                if (checkMovementR(d)) {
                    bufR.push_back(d);
                    lastDataR = d;
                }
            }
        }
        if (bufR.size() > 300) { 
            currentState = RECEIVING_LEFT;
            HC12.write(CMD_STOP); 
        } 
    }

    if (currentState == RECEIVING_LEFT) {
        while (HC12.available()) {
            uint8_t header = HC12.read();
            if (header == CMD_DATA) {
                GloveData temp;
                if (HC12.readBytes((uint8_t*)&temp, sizeof(GloveData)) == sizeof(GloveData)) {
                    bufL.push_back(temp);
                }
            } 
            else if (header == CMD_END) {
                Serial.println("Left Data Received. Processing...");
                processAndPrint();
                currentState = IDLE; 
                break;
            }
        }
    }

    if (currentState != RECEIVING_LEFT && HC12.available()) {
        uint8_t sig = HC12.read();
        if (sig == SIG_CANCEL) {
            if (currentState == RECORDING) {
                currentState = IDLE;
                bufL.clear(); bufR.clear();
                Serial.println("CANCEL_SIGNAL (Remote)");
                blinkLED(2, 200);
            } else {
                Serial.println("DELETE_SIGNAL (Remote)");
                blinkLED(4, 50);
            }
        }
    }
}
