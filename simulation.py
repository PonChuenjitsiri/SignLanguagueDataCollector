import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ==========================================
# 1. ตั้งค่าและโหลดไฟล์ CSV
# ==========================================
CSV_FILE = 'dataset_cf/sim/pon_sim_022326_004.csv'  # อิงจากไฟล์ที่คุณเพิ่งอัปโหลดมาล่าสุด
df = pd.read_csv(CSV_FILE)

# ตั้งค่า Figure สำหรับ 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("Both Hands 3D Simulation from CSV")

# ==========================================
# 2. ฟังก์ชันคณิตศาสตร์ 3D
# ==========================================
def euler_to_matrix(roll, pitch, yaw):
    """แปลงมุมเอียง (Euler) เป็นเมทริกซ์การหมุน 3D"""
    Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 1], [0, 0, 1]])
    return Rz @ Ry @ Rx

def get_hand_model(accel_x, accel_y, accel_z, flex_raw, is_right=False, offset=np.array([0, 0, 0])):
    """สร้างพิกัด 3D ของฝ่ามือและนิ้ว โดยมี Offset เพื่อแยกซ้ายขวา"""
    pitch = np.arctan2(-accel_x, np.sqrt(accel_y**2 + accel_z**2))
    roll = np.arctan2(accel_y, accel_z)
    yaw = 0  
    
    R = euler_to_matrix(roll, pitch, yaw)

    # พิกัดฝ่ามือ (สี่เหลี่ยม)
    palm = np.array([
        [-1, 0, 0],  
        [ 1, 0, 0],  
        [ 1, 2, 0],  
        [-1, 2, 0],  
        [-1, 0, 0]   
    ])
    
    # หมุนและเลื่อนตำแหน่งมือ (Offset)
    palm_rotated = (R @ palm.T).T + offset

    # พิกัดโคนนิ้วทั้ง 5 (กลับด้านมือขวาให้สมจริง)
    if is_right:
        finger_bases = np.array([
            [ 1.2, 1.0, 0], # โป้งขวาอยู่ขวา
            [ 0.8, 2.0, 0], # ชี้
            [ 0.3, 2.0, 0], # กลาง
            [-0.3, 2.0, 0], # นาง
            [-0.8, 2.0, 0]  # ก้อย
        ])
    else:
        finger_bases = np.array([
            [-1.2, 1.0, 0], # โป้งซ้ายอยู่ซ้าย
            [-0.8, 2.0, 0], # ชี้
            [-0.3, 2.0, 0], # กลาง
            [ 0.3, 2.0, 0], # นาง
            [ 0.8, 2.0, 0]  # ก้อย
        ])
        
    finger_bases_rotated = (R @ finger_bases.T).T + offset
    
    # จำลองการงอนิ้ว
    fingers_rotated = []
    for i in range(5):
        # *หมายเหตุ: หากตอน Calibrate สเกลเกิน 2000 สามารถปรับตัวเลขตรงนี้ได้
        bend_angle = np.clip(flex_raw[i] / 2000.0 * (np.pi/1.5), 0, np.pi/1.5) 
        
        finger_vec = np.array([0, np.cos(bend_angle), -np.sin(bend_angle)])
        finger_vec_rotated = R @ finger_vec * 1.5 
        
        tip = finger_bases_rotated[i] + finger_vec_rotated
        fingers_rotated.append((finger_bases_rotated[i], tip))

    return palm_rotated, fingers_rotated

# ==========================================
# 3. ฟังก์ชันอัปเดต Animation
# ==========================================
def update(frame):
    ax.clear()
    
    # ขยายขอบเขตแกน X เพื่อให้วางได้ 2 มือ
    ax.set_xlim([-5, 5])
    ax.set_ylim([-3, 3])
    ax.set_zlim([-3, 3])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    row = df.iloc[frame]
    
    # ดึงข้อมูลมือซ้าย
    L_ax, L_ay, L_az = row['L_Ax'], row['L_Ay'], row['L_Az']
    L_flex = [row['L_F1'], row['L_F2'], row['L_F3'], row['L_F4'], row['L_F5']]
    
    # ดึงข้อมูลมือขวา
    R_ax, R_ay, R_az = row['R_Ax'], row['R_Ay'], row['R_Az']
    R_flex = [row['R_F1'], row['R_F2'], row['R_F3'], row['R_F4'], row['R_F5']]
    
    # คำนวณพิกัด 3D โดยเลื่อนมือซ้ายไป X=-2.5 และเลื่อนมือขวาไป X=+2.5
    L_palm, L_fingers = get_hand_model(L_ax, L_ay, L_az, L_flex, is_right=False, offset=np.array([-2.5, 0, 0]))
    R_palm, R_fingers = get_hand_model(R_ax, R_ay, R_az, R_flex, is_right=True, offset=np.array([ 2.5, 0, 0]))
    
    # วาดฝ่ามือซ้าย (สีน้ำเงิน) และขวา (สีแดงเข้ม)
    ax.plot(L_palm[:,0], L_palm[:,1], L_palm[:,2], color='blue', linewidth=3)
    ax.plot(R_palm[:,0], R_palm[:,1], R_palm[:,2], color='darkred', linewidth=3)
    
    # วาดนิ้วมือ
    colors = ['red', 'green', 'orange', 'purple', 'cyan']
    for i in range(5):
        # นิ้วมือซ้าย
        lb_base, lb_tip = L_fingers[i]
        ax.plot([lb_base[0], lb_tip[0]], [lb_base[1], lb_tip[1]], [lb_base[2], lb_tip[2]], color=colors[i], linewidth=4, marker='o')
        
        # นิ้วมือขวา
        rb_base, rb_tip = R_fingers[i]
        ax.plot([rb_base[0], rb_tip[0]], [rb_base[1], rb_tip[1]], [rb_base[2], rb_tip[2]], color=colors[i], linewidth=4, marker='^')

    ax.text2D(0.05, 0.95, f"Frame: {frame}/{len(df)}", transform=ax.transAxes)

# เล่น Animation
ani = FuncAnimation(fig, update, frames=len(df), interval=20, blit=False)
plt.show()