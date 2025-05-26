import numpy as np
import cv2
from primesense import openni2

# Ruta al directorio donde está OpenNI2.dll
openni2_path = "C:/Orbbec_OpenNI_v2.3.0.86-beta6_windows_release/OpenNI_2.3.0.86_202210111950_4c8f5aa4_beta6_windows/Win64-Release/tools/NiViewer/"
openni2.initialize(openni2_path)  # Inicializa con la ruta a OpenNI2.dll

# Abrir el dispositivo
dev = openni2.Device.open_any()

# Crear flujo de profundidad
depth_stream = dev.create_depth_stream()
depth_stream.start()

print("🎥 Cámara Orbbec conectada. Presiona 'q' para salir.")

while True:
    # Leer un frame
    frame = depth_stream.read_frame()
    frame_data = frame.get_buffer_as_uint16()
    
    # Convertir a matriz NumPy
    depth_array = np.ndarray((frame.height, frame.width), dtype=np.uint16, buffer=frame_data)

    # Normalizar para mostrar
    depth_normalized = cv2.convertScaleAbs(depth_array, alpha=0.03)

    # Mostrar imagen
    cv2.imshow("🌊 Mapa de Profundidad - Orbbec", depth_normalized)

    # Salir al presionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cerrar
depth_stream.stop()
openni2.unload()
cv2.destroyAllWindows()
print("✅ Cámara cerrada.")
