import cv2
import numpy as np
from ultralytics import YOLO
import os
import torch
from openni import openni2
import time

# === Inicializar OpenNI2 ===
openni2.initialize("C:/Orbbec_OpenNI_v2.3.0.86-beta6_windows_release/OpenNI_2.3.0.86_202210111950_4c8f5aa4_beta6_windows/Win64-Release/tools/NiViewer")
dev = openni2.Device.open_any()
depth_stream = dev.create_depth_stream()
depth_stream.start()
print("✅ Cámara de profundidad inicializada.")

# === Esperar a que cámara RGB esté lista ===
cap = cv2.VideoCapture(1)
max_tries = 50
while not cap.isOpened() and max_tries > 0:
    print("⏳ Esperando cámara RGB...")
    cap.open(1)
    time.sleep(0.1)
    max_tries -= 1

if not cap.isOpened():
    print("❌ No se pudo acceder a la cámara RGB.")
    depth_stream.stop()
    openni2.unload()
    exit()
print("✅ Cámara RGB inicializada.")

# === Cargar calibración ===
with np.load("C:/BRAVO/flask name.v2i.yolov12/camera_calibration2.npz") as X:
    mtx, dist = X['mtx'], X['dist']
fx, fy = mtx[0, 0], mtx[1, 1]
cx, cy = mtx[0, 2], mtx[1, 2]

# === Cargar modelo YOLO ===
model = YOLO("C:/BRAVO/flask name.v2i.yolov12/runs/train/EVO7-0.6824/weights/best.pt")

# === Loop principal ===
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame no capturado.")
        continue

    annotated = frame.copy()
    results = model.predict(source=frame, device=0, conf=0.75, stream=False)

    for r in results:
        boxes = r.boxes
        if len(boxes) > 0:
            annotated = r.plot()

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx2d = int((x1 + x2) / 2)
            cy2d = int((y1 + y2) / 2)

            # Obtener profundidad
            depth_frame = depth_stream.read_frame()
            depth_data = np.ndarray((depth_frame.height, depth_frame.width), dtype=np.uint16,
                                    buffer=depth_frame.get_buffer_as_uint16())

            patch = depth_data[max(0, cy2d - 2):cy2d + 3, max(0, cx2d - 2):cx2d + 3]
            valid = patch[patch > 0]
            if valid.size == 0:
                continue

            depth_mm = np.median(valid)
            Z = depth_mm / 10.0
            X_coord = (cx2d - cx) * Z / fx
            Y_coord = -(cy2d - cy) * Z / fy

            print(f"🧭 Objeto detectado (X, Y, Z) = ({X_coord:.2f}, {Y_coord:.2f}, {Z:.2f}) cm")

            cv2.circle(annotated, (cx2d, cy2d), 5, (0, 255, 0), -1)
            cv2.putText(annotated, f"({X_coord:.1f},{Y_coord:.1f},{Z:.1f})cm", (cx2d + 5, cy2d - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("📸 Detección 3D", annotated)
    if 'depth_data' in locals():
        depth_visual = cv2.convertScaleAbs(depth_data, alpha=0.03)
        cv2.imshow("🌊 Mapa de Profundidad", depth_visual)

    if 'depth_visual' in locals():
        cv2.imshow("🌊 Mapa de Profundidad", depth_visual)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === Liberar todo ===
cap.release()
depth_stream.stop()
openni2.unload()
cv2.destroyAllWindows()
