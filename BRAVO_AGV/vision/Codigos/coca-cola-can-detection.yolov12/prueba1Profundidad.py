from ultralytics import YOLO
import cv2
import numpy as np
from openni import openni2
import torch

# Inicializar cámara
openni2.initialize("C:/Orbbec_OpenNI_v2.3.0.86-beta6_windows_release/OpenNI_2.3.0.86_202210111950_4c8f5aa4_beta6_windows/Win64-Release/tools/NiViewer")  # Ruta a tu SDK
dev = openni2.Device.open_any()
depth_stream = dev.create_depth_stream()
depth_stream.start()

cap = cv2.VideoCapture(0)

# Cargar modelo YOLOv8-seg
model = YOLO("yolov8n-seg.pt")  # o tu modelo personalizado de segmentación

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Capturar profundidad
    depth_frame = depth_stream.read_frame()
    depth_data = np.ndarray((depth_frame.height, depth_frame.width), dtype=np.uint16,
                            buffer=depth_frame.get_buffer_as_uint16())

    # Preprocesar imagen de profundidad para visualizar
    depth_viz = cv2.convertScaleAbs(depth_data, alpha=0.03)
    depth_viz = cv2.applyColorMap(depth_viz, cv2.COLORMAP_JET)
    depth_viz = cv2.flip(depth_viz, 1)

    # YOLOv8 segmentación
    results = model.predict(source=frame, conf=0.5, task="segment", verbose=False)

    for r in results:
        if r.masks is None:
            continue

        masks = r.masks.data.cpu().numpy()
        boxes = r.boxes.xyxy.cpu().numpy()

        for i, mask in enumerate(masks):
            # Convertir máscara binaria
            binary_mask = (mask > 0.5).astype(np.uint8)

            # Filtrar profundidad dentro de la máscara
            depth_inside = depth_data[binary_mask == 1]
            valid_depths = depth_inside[depth_inside > 0]

            if len(valid_depths) == 0:
                continue

            depth_mm = np.median(valid_depths)
            Z = depth_mm / 10.0  # en cm

            # Calcular centro del objeto
            x1, y1, x2, y2 = map(int, boxes[i])
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Mostrar en pantalla
            cv2.putText(frame, f"Z: {Z:.1f}cm", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.circle(depth_viz, (cx, cy), 4, (255, 255, 255), -1)
            cv2.putText(depth_viz, "Z centro", (cx + 5, cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Mostrar ambas vistas
    cv2.imshow("RGB + Anotaciones", frame)
    cv2.imshow("Z visual", depth_viz)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
depth_stream.stop()
openni2.unload()
cv2.destroyAllWindows()
