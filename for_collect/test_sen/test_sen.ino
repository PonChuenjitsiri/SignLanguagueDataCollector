#include <Arduino.h>
#include <Wire.h>
#include <MPU9250_asukiaaa.h>

// --- Config Pins ---
#define PIN_BUTTON 5
#define PIN_LED    10  

// --- Default Thresholds ---
const int T_FLEX_DEFAULT = 300; 
int t_flex = T_FLEX_DEFAULT;

// เก็บ Threshold แยกแกน [0]=X, [1]=Y, [2]=Z
int t_accel[3] = {30, 30, 30};      
int t_gyro[3]  = {2000, 2000, 2000}; 

// --- Calibration Variables ---
int flexMin[5] = {0, 0, 0, 0, 0};      
int flexMax[5] = {1023, 1023, 1023, 1023, 1023};
bool isCalibrated = false;

struct GloveData {
    uint16_t flex[5];
    int16_t accel[3];
    int16_t gyro[3];
};

MPU9250_asukiaaa mpu;
GloveData lastData; 
bool firstRun = true;
unsigned long lastHeartbeat = 0; 

// --- Button State Variables ---
unsigned long buttonPressStartTime = 0;
bool isButtonHeld = false;
bool actionTriggered = false;

// --- Helper Functions ---

void blinkLED(int times, int delayTime) {
    for(int i=0; i<times; i++) {
        digitalWrite(PIN_LED, HIGH);
        delay(delayTime);
        digitalWrite(PIN_LED, LOW);
        delay(delayTime);
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
    while(digitalRead(PIN_BUTTON) == HIGH) delay(10); // รอปล่อย
    delay(100); 
    while(digitalRead(PIN_BUTTON) == LOW) delay(10);  // รอกดใหม่
    delay(100); 
    while(digitalRead(PIN_BUTTON) == HIGH) delay(10); // รอปล่อย
    delay(100); 
}

void printHeader() {
    Serial.println("\n--- SENSOR MONITOR MODE ---");
    Serial.print("Thres Accel(X,Y,Z): "); 
    Serial.print(t_accel[0]); Serial.print(","); Serial.print(t_accel[1]); Serial.print(","); Serial.println(t_accel[2]);
    Serial.print("Thres Gyro(X,Y,Z): "); 
    Serial.print(t_gyro[0]); Serial.print(","); Serial.print(t_gyro[1]); Serial.print(","); Serial.println(t_gyro[2]);
    
    Serial.println("Format: [Time] Trigger | Flex(0-1000) | Accel | Gyro");
    Serial.println("-----------------------------------------------------------");
}

void printData(String trigger, GloveData d) {
    Serial.print("["); Serial.print(millis()); Serial.print("] Trg: ");
    Serial.print(trigger); 
    
    while(trigger.length() < 12) { Serial.print(" "); trigger += " "; }

    Serial.print("| F: ");
    for(int i=0; i<5; i++) {
        Serial.print(d.flex[i]);
        Serial.print(" ");
    }

    Serial.print("| A: ");
    Serial.print(d.accel[0]); Serial.print(",");
    Serial.print(d.accel[1]); Serial.print(",");
    Serial.print(d.accel[2]);

    Serial.print(" | G: ");
    Serial.print(d.gyro[0]); Serial.print(",");
    Serial.print(d.gyro[1]); Serial.print(",");
    Serial.print(d.gyro[2]);
    Serial.println();
}

void calibrateSensors() {
    Serial.println("\n=== PART 1: FLEX SENSOR CALIBRATION ===");
    blinkLED(5, 100); 
    digitalWrite(PIN_LED, HIGH); 

    long sumOpen[5] = {0,0,0,0,0};
    long sumClose[5] = {0,0,0,0,0};

    // --- Flex Calibration ---
    for (int round = 1; round <= 5; round++) {
        Serial.print(">> ROUND "); Serial.print(round); Serial.println(" / 5");

        Serial.println("   [ACTION] OPEN hand -> Press Button");
        waitForUserAction(); 
        
        digitalWrite(PIN_LED, LOW);
        Serial.print("    Reading Open...");
        for(int i=0; i<5; i++) sumOpen[i] += analogRead(i);
        Serial.println(" Done.");

        Serial.println("   [ACTION] CLOSE hand -> Press Button");
        waitForUserAction(); 
        
        Serial.print("    Reading Close...");
        for(int i=0; i<5; i++) sumClose[i] += analogRead(i);
        Serial.println(" Done.");

        blinkLED(2, 100);
        if(round < 5) digitalWrite(PIN_LED, HIGH);
    }

    for(int i=0; i<5; i++) {
        flexMin[i] = sumOpen[i] / 5;
        flexMax[i] = sumClose[i] / 5;
        if (abs(flexMax[i] - flexMin[i]) < 10) flexMax[i] = flexMin[i] + 100; 
    }
    
    // --- PART 2: IMU CALIBRATION (Separate Axis) ---
    Serial.println("\n=== PART 2: IMU NOISE CALIBRATION (Individual Axis) ===");
    Serial.println("   [INSTRUCTION] KEEP HAND STILL FOR 4 SECONDS...");
    
    digitalWrite(PIN_LED, HIGH); 
    delay(200); 
    
    // ตัวแปรเก็บ Max Noise แยกแกน
    long maxDiffAccel[3] = {0, 0, 0};
    long maxDiffGyro[3]  = {0, 0, 0};
    
    GloveData prevIMU;
    readMPU(prevIMU); 
    
    unsigned long startCalib = millis();
    int sampleCount = 0;

    while(millis() - startCalib < 4000) {
        GloveData currIMU;
        readMPU(currIMU);

        // เช็ค Noise แยกแกน X, Y, Z
        for(int k=0; k<3; k++) {
            long diffA = abs(currIMU.accel[k] - prevIMU.accel[k]);
            if(diffA > maxDiffAccel[k]) maxDiffAccel[k] = diffA;

            long diffG = abs(currIMU.gyro[k] - prevIMU.gyro[k]);
            if(diffG > maxDiffGyro[k]) maxDiffGyro[k] = diffG;
        }

        prevIMU = currIMU;
        sampleCount++;
        delay(10); 
    }
    
    digitalWrite(PIN_LED, LOW); 
    Serial.print("   Samples: "); Serial.println(sampleCount);

    // --- Calculate New Thresholds ---
    Serial.println("   Calculated Noise Levels (Max Delta):");
    Serial.print("   Accel: "); 
    Serial.print(maxDiffAccel[0]); Serial.print(", ");
    Serial.print(maxDiffAccel[1]); Serial.print(", ");
    Serial.println(maxDiffAccel[2]);
    
    Serial.print("   Gyro:  "); 
    Serial.print(maxDiffGyro[0]); Serial.print(", ");
    Serial.print(maxDiffGyro[1]); Serial.print(", ");
    Serial.println(maxDiffGyro[2]);

    // Set Thresholds (Noise * 1.5 + Min Limit)
    for(int k=0; k<3; k++) {
        // Accel
        t_accel[k] = (int)(maxDiffAccel[k] * 1.5);
        if (t_accel[k] < 20) t_accel[k] = 20; // Min Limit 0.20

        // Gyro
        t_gyro[k] = (int)(maxDiffGyro[k] * 1.5);
        if (t_gyro[k] < 500) t_gyro[k] = 500; // Min Limit 5.00
    }

    Serial.println("\n=== CALIBRATION COMPLETED ===");
    isCalibrated = true;
    blinkLED(3, 200);
}

void setup() {
    Serial.begin(115200);
    pinMode(PIN_BUTTON, INPUT); 
    pinMode(PIN_LED, OUTPUT);
    
    Wire.begin(6, 7); 
    mpu.setWire(&Wire);
    mpu.beginAccel();
    mpu.beginGyro();

    printHeader();
    delay(1000);
}

void loop() {
    // --- 1. Button Logic ---
    int btnState = digitalRead(PIN_BUTTON);
    if (btnState == HIGH) { 
        if (!isButtonHeld) {
            isButtonHeld = true;
            buttonPressStartTime = millis();
            actionTriggered = false; 
        } else {
            unsigned long duration = millis() - buttonPressStartTime;
            if (duration > 3000 && !actionTriggered) {
                actionTriggered = true; 
                calibrateSensors();     
                printHeader(); 
                firstRun = true; 
                isButtonHeld = false; 
                return; 
            }
        }
    } else {
        isButtonHeld = false;
        actionTriggered = false;
    }

    // --- 2. Read Sensors ---
    GloveData current;
    readMPU(current);

    for(int i=0; i<5; i++) {
        int raw = analogRead(i);
        if (isCalibrated) {
            current.flex[i] = map(constrain(raw, min(flexMin[i], flexMax[i]), max(flexMin[i], flexMax[i])), 
                                  flexMin[i], flexMax[i], 
                                  0, 1000);
        } else {
            current.flex[i] = raw;
        }
    }

    // --- 3. Monitor Logic (Separate Thresholds) ---
    if (firstRun) {
        lastData = current;
        printData("FirstRun", current);
        firstRun = false;
        return;
    }

    String triggerSource = "";
    bool isMoving = false;

    // Check Flex
    for(int i=0; i<5; i++) {
        if (abs((int)current.flex[i] - (int)lastData.flex[i]) > t_flex) {
            isMoving = true; triggerSource += "F" + String(i) + " "; 
        }
    }
    
    // Check Accel (เทียบแต่ละแกนกับ Threshold ของแกนนั้นๆ)
    if (abs(current.accel[0] - lastData.accel[0]) > t_accel[0]) { isMoving = true; triggerSource += "AX "; }
    if (abs(current.accel[1] - lastData.accel[1]) > t_accel[1]) { isMoving = true; triggerSource += "AY "; }
    if (abs(current.accel[2] - lastData.accel[2]) > t_accel[2]) { isMoving = true; triggerSource += "AZ "; }

    // Check Gyro (เทียบแต่ละแกนกับ Threshold ของแกนนั้นๆ)
    if (abs(current.gyro[0] - lastData.gyro[0]) > t_gyro[0]) { isMoving = true; triggerSource += "GX "; }
    if (abs(current.gyro[1] - lastData.gyro[1]) > t_gyro[1]) { isMoving = true; triggerSource += "GY "; }
    if (abs(current.gyro[2] - lastData.gyro[2]) > t_gyro[2]) { isMoving = true; triggerSource += "GZ "; }

    if (isMoving) {
        printData(triggerSource, current);
        lastData = current; 
        lastHeartbeat = millis(); 
    } else {
        if (millis() - lastHeartbeat > 3000) {
            Serial.print("."); 
            lastHeartbeat = millis();
        }
    }
    
    delay(20); 
}
