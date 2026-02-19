#include <Arduino.h>
#include <Wire.h>
#include <MPU9250_asukiaaa.h>
#include <vector>

HardwareSerial HC12(1);
#define HC12_RX 20
#define HC12_TX 21
#define PIN_BUTTON 5
#define PIN_LED 10 

int t_flex = 100; 
int t_accel[3] = {25, 25, 25};      
int t_gyro[3]  = {2000, 2000, 2000}; 

int flexMin[5] = {0, 0, 0, 0, 0};      
int flexMax[5] = {4095, 4095, 4095, 4095, 4095};
bool isCalibrated = false;

struct GloveData {
    uint16_t flex[5];
    int16_t accel[3];
    int16_t gyro[3];
};

MPU9250_asukiaaa mpu;
GloveData lastData;
std::vector<GloveData> storage; 
bool isRecording = false;

const uint8_t CMD_START = 0xA1;
const uint8_t CMD_STOP  = 0xA2;
const uint8_t CMD_DATA  = 0xD1;
const uint8_t CMD_END   = 0xD2;
const uint8_t SIG_CANCEL = 0xEE;

unsigned long btnPressStart = 0;
bool isBtnHeld = false;
bool actionTriggered = false;

void blinkLED(int times, int duration) {
    pinMode(PIN_LED, OUTPUT);
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
    Serial.println("\n=== CALIBRATION MODE (LEFT HAND) ===");
    blinkLED(5, 100);

    long sumOpen[5] = {0,0,0,0,0};
    long sumClose[5] = {0,0,0,0,0};

    for (int round = 1; round <= 5; round++) {
        Serial.printf(">> ROUND %d/5\n", round);
        waitForUserAction();
        for(int i=0; i<5; i++) sumOpen[i] += analogRead(i); 
        
        waitForUserAction();
        for(int i=0; i<5; i++) sumClose[i] += analogRead(i); 
        
        blinkLED(2, 100);
    }

    for(int i=0; i<5; i++) {
        flexMin[i] = sumOpen[i] / 5;
        flexMax[i] = sumClose[i] / 5;
        if (flexMin[i] == flexMax[i]) flexMax[i] += 1; // Safety
    }

    Serial.println("Keep Still 4s...");
    blinkLED(1, 1000); 

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

    for(int k=0; k<3; k++) {
        t_accel[k] = max((int)(maxDiffAccel[k] * 1.5), 20);
        t_gyro[k]  = max((int)(maxDiffGyro[k] * 1.5), 500);
    }
    
    isCalibrated = true;
    Serial.println("DONE");
    blinkLED(3, 200);

    while(digitalRead(PIN_BUTTON) == HIGH) delay(10);
    isBtnHeld = false;
    actionTriggered = false;
}

bool checkMovement(GloveData current) {
    if (storage.empty()) return true;
    for(int i=0; i<5; i++) if (abs((int)current.flex[i] - (int)lastData.flex[i]) > t_flex) return true;
    for(int k=0; k<3; k++) {
        if (abs(current.accel[k] - lastData.accel[k]) > t_accel[k]) return true;
        if (abs(current.gyro[k] - lastData.gyro[k]) > t_gyro[k]) return true;
    }
    return false;
}

void setup() {
    Serial.begin(115200);
    HC12.begin(115200, SERIAL_8N1, HC12_RX, HC12_TX);
    analogReadResolution(12); // ชัดเจนว่า 0-4095
    pinMode(PIN_BUTTON, INPUT);
    pinMode(PIN_LED, OUTPUT); 
    Wire.begin(6, 7);
    mpu.setWire(&Wire); mpu.beginAccel(); mpu.beginGyro();
    Serial.println("--- LEFT HAND READY ---");
}

void sendDataToMaster() {
    Serial.print("Sending data: "); Serial.println(storage.size());
    for (size_t i = 0; i < storage.size(); i++) {
        HC12.write(CMD_DATA);
        HC12.write((uint8_t*)&storage[i], sizeof(GloveData));
        delay(10);
    }
    HC12.write(CMD_END);
    Serial.println("Send Complete");
    storage.clear();
}

void loop() {
    if (HC12.available()) {
        uint8_t cmd = HC12.read();
        if (cmd == CMD_START) {
            isRecording = true;
            storage.clear();
            memset(&lastData, 0, sizeof(GloveData));
            Serial.println("CMD: START");
        } 
        else if (cmd == CMD_STOP) {
            isRecording = false;
            Serial.println("CMD: STOP -> Sending Data...");
            sendDataToMaster();
        }
    }

    if (isRecording) {
        static uint32_t last_scan = 0;
        if (millis() - last_scan >= 20) { 
            last_scan = millis();
            if (mpu.accelUpdate() == 0 && mpu.gyroUpdate() == 0) {
                GloveData d;
                readMPU(d);
                for(int i=0; i<5; i++) {
                    int raw = analogRead(i);
                    if (isCalibrated) {
                        // 1. Constrain: ตัดค่าเกินทิ้ง
                        int clipped = constrain(raw, min(flexMin[i], flexMax[i]), max(flexMin[i], flexMax[i]));
                        // 2. Map: บังคับช่วง 0-1000
                        d.flex[i] = map(clipped, flexMin[i], flexMax[i], 0, 1000);
                    } else {
                        d.flex[i] = raw;
                    }
                }
                if (checkMovement(d)) {
                    storage.push_back(d);
                    lastData = d;
                }
            }
        }
    }

    int btnState = digitalRead(PIN_BUTTON);
    if (btnState == HIGH) { // กด
        if (!isBtnHeld) {
            isBtnHeld = true;
            btnPressStart = millis();
            actionTriggered = false;
        } else {
            if (millis() - btnPressStart > 3000 && !actionTriggered) {
                if (!isRecording) { 
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
                HC12.write(SIG_CANCEL);
                Serial.println("Sent CANCEL/DELETE Signal");
                isRecording = false;
                storage.clear();
            }
            isBtnHeld = false;
        }
    }
}
