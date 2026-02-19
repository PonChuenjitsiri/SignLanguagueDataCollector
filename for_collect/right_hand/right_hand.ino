#include <Arduino.h>
#include <Wire.h>
#include <MPU9250_asukiaaa.h>
#include <vector>

HardwareSerial HC12(1);
#define HC12_RX 20
#define HC12_TX 21

// --- Config Pins ---
const int PIN_LED = 10;
const int PIN_BUTTON = 5;

// --- Calibration & Thresholds ---
int t_flex = 100; 
int t_accel[3] = {25, 25, 25};      
int t_gyro[3]  = {2000, 2000, 2000}; 

// ค่า Default ก่อน Calibrate (เผื่อยังไม่ได้ทำ)
int flexMin[5] = {0, 0, 0, 0, 0};      
int flexMax[5] = {4095, 4095, 4095, 4095, 4095}; // ESP32 อ่านได้สูงสุด 4095
bool isCalibrated = false;

struct GloveData {
    uint16_t flex[5];
    int16_t accel[3];
    int16_t gyro[3];
};

MPU9250_asukiaaa mpu;

// Protocol Bytes
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

// --- Helper Functions ---
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
    while(digitalRead(PIN_BUTTON) == HIGH) delay(10);  // รอปล่อย
    delay(100); 
    while(digitalRead(PIN_BUTTON) == LOW) delay(10);   // รอกดใหม่
    delay(100); 
    while(digitalRead(PIN_BUTTON) == HIGH) delay(10);  // รอปล่อย
    delay(100); 
}

void calibrateSensors() {
    Serial.println("\n=== CALIBRATION MODE (RIGHT HAND) ===");
    blinkLED(5, 100); digitalWrite(PIN_LED, HIGH);

    long sumOpen[5] = {0,0,0,0,0};
    long sumClose[5] = {0,0,0,0,0};

    // 1. Flex Calibration (เก็บค่าดิบ 0-4095)
    for (int round = 1; round <= 5; round++) {
        Serial.printf(">> ROUND %d/5\n", round);
        
        Serial.println("   [ACTION] OPEN hand (Read Min) -> Press Button");
        waitForUserAction();
        digitalWrite(PIN_LED, LOW);
        for(int i=0; i<5; i++) sumOpen[i] += analogRead(4-i); 
        Serial.println("    Read Open Done.");

        Serial.println("   [ACTION] CLOSE hand (Read Max) -> Press Button");
        waitForUserAction();
        for(int i=0; i<5; i++) sumClose[i] += analogRead(4-i); 
        Serial.println("    Read Close Done.");

        blinkLED(2, 100);
        if(round < 5) digitalWrite(PIN_LED, HIGH);
    }

    // คำนวณค่า Min/Max เฉลี่ย
    Serial.println("Flex Sensor Mapping (Raw -> 0-1000):");
    for(int i=0; i<5; i++) {
        flexMin[i] = sumOpen[i] / 5;
        flexMax[i] = sumClose[i] / 5;
        
        // Safety: ป้องกันกรณีค่าเท่ากัน (หาร 0)
        if (flexMin[i] == flexMax[i]) flexMax[i] += 1; 
        
        Serial.printf("  F%d Raw Range: %d - %d -> Will map to 0 - 1000\n", i, flexMin[i], flexMax[i]);
    }

    // 2. IMU Calibration
    Serial.println("\n=== IMU NOISE CALIBRATION (Keep Still 4s) ===");
    digitalWrite(PIN_LED, HIGH); delay(500);

    long maxDiffAccel[3] = {0,0,0};
    long maxDiffGyro[3] = {0,0,0};
    GloveData prev, curr;
    readMPU(prev);
    
    unsigned long startT = millis();
    while(millis() - startT < 4000) {
        readMPU(curr);
        for(int k=0; k<3; k++) {
            long diffA = abs(curr.accel[k] - prev.accel[k]);
            if(diffA > maxDiffAccel[k]) maxDiffAccel[k] = diffA;
            long diffG = abs(curr.gyro[k] - prev.gyro[k]);
            if(diffG > maxDiffGyro[k]) maxDiffGyro[k] = diffG;
        }
        prev = curr;
        delay(10);
    }
    digitalWrite(PIN_LED, LOW);

    for(int k=0; k<3; k++) {
        t_accel[k] = max((int)(maxDiffAccel[k] * 1.5), 20);
        t_gyro[k]  = max((int)(maxDiffGyro[k] * 1.5), 500);
    }
    
    isCalibrated = true;
    Serial.println("=== CALIBRATION DONE ===");
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
    for(int k=0; k<3; k++) {
        if (abs(current.accel[k] - lastDataR.accel[k]) > t_accel[k]) return true;
        if (abs(current.gyro[k] - lastDataR.gyro[k]) > t_gyro[k]) return true;
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
    
    // ตั้งค่าความละเอียด ADC เป็น 12-bit (0-4095) ให้ชัดเจน
    analogReadResolution(12);
    
    pinMode(PIN_BUTTON, INPUT); 
    pinMode(PIN_LED, OUTPUT);
    Wire.begin(6, 7);
    mpu.setWire(&Wire); mpu.beginAccel(); mpu.beginGyro();
    Serial.println("\n--- MASTER (RIGHT HAND) READY ---");
}

void loop() {
    int btnState = digitalRead(PIN_BUTTON); 
    if (btnState == HIGH) { // กด
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
    } else { // ปล่อย
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
                    int raw = analogRead(4-i); // ขวาใช้ 4-i
                    if (isCalibrated) {
                        // 1. Constrain: บังคับค่าให้อยู่ในช่วงที่ Calibrate เท่านั้น (ตัด Noise ที่เกินขอบเขต)
                        int clipped = constrain(raw, min(flexMin[i], flexMax[i]), max(flexMin[i], flexMax[i]));
                        // 2. Map: แปลงช่วงที่ตัดแล้ว ให้เป็น 0-1000 เป๊ะๆ
                        d.flex[i] = map(clipped, flexMin[i], flexMax[i], 0, 1000);
                    } else {
                        d.flex[i] = raw; // ถ้ายังไม่ Calibrate ให้ใช้ค่าดิบ 0-4095 ไปก่อน
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
