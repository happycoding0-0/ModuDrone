# -*- coding: utf-8 -*-
import cv2
import numpy as np
import math
from time import sleep
from ultralytics import YOLO
from collections import deque

# --- 설정 (Configuration) ---
MODEL_PATH = 'yolov8s.pt'
GRID_X, GRID_Y = 4, 3

# --- 드론 및 카메라 설정 (중요) ---
# 이 값을 실제 비행 환경에 맞게 수정해야 정확한 밀도 계산이 가능합니다.
DRONE_ALTITUDE_METERS = 20  # 드론의 비행 고도 (미터 단위)
CAMERA_FOV_DEGREES = 84     # 드론 카메라의 수평 화각 (도 단위)

# --- 밀도 기반 위험 기준 (명/㎡) ---
DANGER_DENSITY = 6.0   # 1㎡ 당 6명 이상: 위험 (빨간색)
WARNING_DENSITY = 4.0  # 1㎡ 당 4명 이상: 주의 (주황색)

# --- HUD 및 기타 설정 ---
history = deque(maxlen=100)

# --- 하드웨어 및 모델 초기화 ---
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("오류: 카메라를 열 수 없습니다.")
    exit()

print("밀도 기반 실시간 분석 시스템을 시작합니다.")
sleep(1)

# --- 메인 루프 (실시간 분석) ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape

    # 1. 실제 면적 계산
    # 삼각함수를 이용해 영상의 실제 가로/세로 길이(m)와 면적(㎡)을 계산
    fov_radians = math.radians(CAMERA_FOV_DEGREES)
    view_width_meters = 2 * DRONE_ALTITUDE_METERS * math.tan(fov_radians / 2)
    view_height_meters = view_width_meters * (H / W)
    total_area_sq_meters = view_width_meters * view_height_meters
    grid_area_sq_meters = total_area_sq_meters / (GRID_X * GRID_Y)

    # 2. YOLO 객체 탐지 및 인원수 카운트
    results = model.predict(frame, classes=[0], conf=0.35, imgsz=640, verbose=False)
    grid_counts = np.zeros((GRID_Y, GRID_X), dtype=int)
    detected_boxes = results[0].boxes.xyxy.cpu().numpy()
    for box in detected_boxes:
        x1, y1, x2, y2 = map(int, box)
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        grid_j, grid_i = min(center_x // (W // GRID_X), GRID_X - 1), min(center_y // (H // GRID_Y), GRID_Y - 1)
        grid_counts[grid_i, grid_j] += 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 128, 0), 2)

    total_people = grid_counts.sum()
    history.append(total_people)
    
    # 3. 밀도 기반 격자 시각화
    max_density = 0.0
    for i in range(GRID_Y):
        for j in range(GRID_X):
            count = grid_counts[i, j]
            if count > 0:
                density = count / grid_area_sq_meters
                max_density = max(max_density, density)
                
                # 밀도에 따른 색상 결정
                if density >= DANGER_DENSITY:
                    color = (0, 0, 255)  # 위험: 빨강
                elif density >= WARNING_DENSITY:
                    color = (0, 165, 255) # 주의: 주황
                else:
                    color = (0, 255, 0)    # 안전: 초록
                
                alpha = 0.4
                overlay = frame.copy()
                cv2.rectangle(overlay, (j * (W//GRID_X), i * (H//GRID_Y)), ((j + 1) * (W//GRID_X), (i + 1) * (H//GRID_Y)), color, -1)
                frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
                
                # 격자에 밀도 텍스트 표시
                text_content = f"{density:.1f}/m2"
                cv2.putText(frame, text_content, (j * (W//GRID_X) + 10, i * (H//GRID_Y) + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # 4. 업그레이드된 HUD
    dashboard = np.zeros((120, W, 3), dtype="uint8")
    frame[H-120:H, 0:W] = cv2.addWeighted(frame[H-120:H, 0:W], 0.3, dashboard, 0.7, 0)
    cv2.putText(frame, f"ALT: {DRONE_ALTITUDE_METERS}m | AREA/GRID: {grid_area_sq_meters:.1f} m2", (15, H - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, f"TOTAL PEOPLE: {total_people}", (15, H - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"MAX DENSITY: {max_density:.1f} p/m2", (15, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow("Real-time Density-based Analysis", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
