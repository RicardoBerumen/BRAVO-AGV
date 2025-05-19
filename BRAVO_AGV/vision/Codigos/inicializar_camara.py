import numpy as np
import cv2
from primesense import openni2

import sys
print("Python en uso:", sys.executable)

#openni2_path = "/home/bravo/Documents/Orbbec_OpenNI_v2.3.0.86-beta6_linux_release/" \
               #"OpenNI_2.3.0.86_202210111154_4c8f5aa4_beta6_linux_x64/" \
               #"OpenNI_2.3.0.86_202210111154_4c8f5aa4_beta6_linux/sdk/libs/"
            #    BRAVO_AGV/vision/Orbbec_OpenNI_v2.3.0.86-beta6_linux_release/OpenNI_2.3.0.86_202210111155_4c8f5aa4_beta6_a311d_arm64/OpenNI_2.3.0.86_202210111155_4c8f5aa4_beta6_a311d/sdk
openni2_path="/home/bravo/bravo_ws/src/BRAVO-AGV/BRAVO_AGV/vision/Orbbec_OpenNI_v2.3.0.86-beta6_linux_release" \
"/OpenNI_2.3.0.86_202210111155_4c8f5aa4_beta6_a311d_arm64/OpenNI_2.3.0.86_202210111155_4c8f5aa4_beta6_a311d/sdk/libs"

openni2.initialize(openni2_path)

dev = openni2.Device.open_any()

# Crear flujo de profundidad
depth_stream = dev.create_depth_stream()
depth_stream.start()

cv2.namedWindow("Mapa de Profundidad - Orbbec", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Mapa de Profundidad - Orbbec", 960, 720)

print("Cámara Orbbec conectada. Presiona 'q' para salir.")

while True:
    # Leer un frame
    frame = depth_stream.read_frame()
    frame_data = frame.get_buffer_as_uint16()

    # Convertir a matriz NumPy
    depth_array = np.ndarray((frame.height, frame.width), dtype=np.uint16, buffer=frame_data)

    depth_normalized = cv2.convertScaleAbs(depth_array, alpha=0.03)
    cv2.imshow("Mapa de Profundidad - Orbbec", depth_normalized)

    # Salir al presionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


depth_stream.stop()
openni2.unload()
cv2.destroyAllWindows()
print("Cámara cerrada.")
