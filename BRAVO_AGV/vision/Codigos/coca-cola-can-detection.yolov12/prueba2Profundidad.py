import cv2
import numpy as np
from openni import openni2

# === Inicializar OpenNI2 ===
openni2.initialize("C:/Orbbec_OpenNI_v2.3.0.86-beta6_windows_release/OpenNI_2.3.0.86_202210111950_4c8f5aa4_beta6_windows/Win64-Release/tools/NiViewer")
dev = openni2.Device.open_any()
depth_stream = dev.create_depth_stream()
depth_stream.start()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("No se pudo abrir la cámara RGB.")
    exit()

# === Crear ventana con sliders ===
cv2.namedWindow("HSV Ajustes")

def nothing(x):
    pass

# Crear sliders para H, S, V
cv2.createTrackbar("H min", "HSV Ajustes", 0, 180, nothing)
cv2.createTrackbar("H max", "HSV Ajustes", 180, 180, nothing)
cv2.createTrackbar("S min", "HSV Ajustes", 201, 255, nothing)
cv2.createTrackbar("S max", "HSV Ajustes", 255, 255, nothing)
cv2.createTrackbar("V min", "HSV Ajustes", 50, 255, nothing)
cv2.createTrackbar("V max", "HSV Ajustes", 255, 255, nothing)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Captura profundidad
    depth_frame = depth_stream.read_frame()
    depth_data = np.ndarray((depth_frame.height, depth_frame.width), dtype=np.uint16,
                            buffer=depth_frame.get_buffer_as_uint16())

    # Convertir a HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Leer valores de sliders
    h_min = cv2.getTrackbarPos("H min", "HSV Ajustes")
    h_max = cv2.getTrackbarPos("H max", "HSV Ajustes")
    s_min = cv2.getTrackbarPos("S min", "HSV Ajustes")
    s_max = cv2.getTrackbarPos("S max", "HSV Ajustes")
    v_min = cv2.getTrackbarPos("V min", "HSV Ajustes")
    v_max = cv2.getTrackbarPos("V max", "HSV Ajustes")

    # Crear máscara con rango ajustado
    lower_bound = np.array([h_min, s_min, v_min], np.uint8)
    upper_bound = np.array([h_max, s_max, v_max], np.uint8)
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Limpiar máscara
    kernel = np.ones((5, 5), np.uint8)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Buscar contornos y centro
    x, y, w, h = cv2.boundingRect(mask_clean)
    cx = x + w // 2
    cy = y + h // 2

    if 0 <= cx < frame.shape[1] and 0 <= cy < frame.shape[0]:
        # === Parche pequeño centrado ===
        patch_size = 7
        half = patch_size // 2
        x_start = max(0, cx - half)
        y_start = max(0, cy - half)
        x_end = min(depth_data.shape[1], cx + half + 1)
        y_end = min(depth_data.shape[0], cy + half + 1)

        # Obtener parche de profundidad y máscara
        patch_depth = depth_data[y_start:y_end, x_start:x_end]
        patch_mask = mask_clean[y_start:y_end, x_start:x_end]
        patch_valid = patch_depth[(patch_mask == 255) & (patch_depth > 0) & (patch_depth < 10000)]

        if len(patch_valid) > 0:
            depth_cm = np.median(patch_valid) / 10.0
            print(f"✔ Z en el centro de la máscara: {depth_cm:.2f} cm")
            cv2.putText(frame, f"Z: {depth_cm:.1f} cm", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            print("❌ Sin profundidad válida en el centro de la máscara.")

        # Mostrar en el Mapa Z
        depth_visual = cv2.convertScaleAbs(depth_data, alpha=0.03)
        depth_visual = cv2.applyColorMap(depth_visual, cv2.COLORMAP_JET)
        depth_visual = cv2.flip(depth_visual, 1)

        # Rectángulo solo en parche central
        cv2.rectangle(depth_visual, (x_start, y_start), (x_end, y_end), (0, 255, 255), 2)
        cv2.putText(depth_visual, "Z centro", (x_start, y_start - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        cv2.imshow("Mapa Z", depth_visual)


    # Mostrar resultado
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

    cv2.imshow("RGB + Z", frame)
    cv2.imshow("Máscara", mask_clean)

    key = cv2.waitKey(1)
    if key == ord('q'):
        print(f"Rango HSV final:")
        print(f"Lower: [{h_min}, {s_min}, {v_min}]")
        print(f"Upper: [{h_max}, {s_max}, {v_max}]")
        break

# === Finalizar ===
cap.release()
depth_stream.stop()
openni2.unload()
cv2.destroyAllWindows()
