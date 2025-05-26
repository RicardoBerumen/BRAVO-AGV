
import cv2
import numpy as np
from pyorbbecsdk import *

ESC_KEY = 27

def main():
    pipeline = Pipeline()
    device = pipeline.get_device()
    config = Config()

    # Obtener perfiles de flujo por defecto
    color_profile = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR).get_default_video_stream_profile()
    depth_profile = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR).get_default_video_stream_profile()

    config.enable_stream(color_profile)
    config.enable_stream(depth_profile)
    config.set_align_mode(OBAlignMode.SW_MODE)  # alineación profundidad-color

    pipeline.start(config)

    while True:
        try:
            frameset = pipeline.wait_for_frames(100)
            color_frame = frameset.get_color_frame()
            depth_frame = frameset.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            # Convertir a imagen BGR
            color_img = np.frombuffer(color_frame.get_data(), dtype=np.uint8).reshape((color_frame.get_height(), color_frame.get_width(), 3))

            depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape((depth_frame.get_height(), depth_frame.get_width()))
            depth_data = depth_data.astype(np.float32) * depth_frame.get_depth_scale() * 100  # a cm

            depth_viz = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            depth_viz = cv2.applyColorMap(depth_viz, cv2.COLORMAP_JET)

            hsv = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, (100, 150, 0), (140, 220, 255))
            mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

            x, y, w, h = cv2.boundingRect(mask_clean)
            cx = x + w // 2
            cy = y + h // 2

            # Medición en parche 7x7 en centro
            half = 3
            x1, y1 = max(0, cx - half), max(0, cy - half)
            x2, y2 = min(depth_data.shape[1], cx + half + 1), min(depth_data.shape[0], cy + half + 1)

            patch = depth_data[y1:y2, x1:x2]
            patch_mask = mask_clean[y1:y2, x1:x2]
            valid = patch[(patch_mask == 255) & (patch > 0) & (patch < 1000)]

            if len(valid) > 0:
                z_cm = np.median(valid)
                cv2.putText(color_img, f"Z: {z_cm:.1f} cm", (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                print(f"Z={z_cm:.1f} cm en ({cx}, {cy})")

            cv2.rectangle(depth_viz, (x1, y1), (x2, y2), (0, 255, 255), 1)
            cv2.putText(depth_viz, "Z centro", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

            cv2.imshow("Z Visual", depth_viz)
            cv2.imshow("RGB", color_img)
            cv2.imshow("Mask", mask_clean)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except KeyboardInterrupt:
            break

    pipeline.stop()

if __name__ == '__main__':
    print("Usando pyorbbecsdk para calcular Z desde el centro de la máscara azul")
    main()