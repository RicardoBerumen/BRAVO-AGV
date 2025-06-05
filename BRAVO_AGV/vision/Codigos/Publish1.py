import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
import numpy as np
import cv2
import torch
from ultralytics import YOLO
import message_filters

class Object3DDetector(Node):
    def __init__(self):
        super().__init__('object_3d_detector')

        # Calibración de cámara
        with np.load("/home/evo/Documents/orbbec/Orbbec_descargas/VISION-BRAVO-main/Codigos/camera_calibration_logi.npz") as X:
            self.mtx, self.dist = X['mtx'], X['dist']
            self.fx, self.fy = self.mtx[0, 0], self.mtx[1, 1]
            self.cx, self.cy = self.mtx[0, 2], self.mtx[1, 2]

        # Corrección de profundidad calibrada (Z real)
        depth_model = np.load("/home/evo/Documents/orbbec/Orbbec_descargas/VISION-BRAVO-main/Codigos/depth_model_fit_yolo.npz")
        self.a = depth_model["a"]
        self.b = depth_model["b"]
        self.c = depth_model["c"]

        # Modelo YOLOv12
        self.model = YOLO("/home/evo/Documents/coca-cola-can-detection.yolov12/runs/train/EVO1/weights/best.pt")
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        self.bridge = CvBridge()

        # Subscripción a imágenes RGB y profundidad comprimidas
        rgb_sub = message_filters.Subscriber(self, CompressedImage, "/camera/camera/color/image_raw/compressed")
        depth_sub = message_filters.Subscriber(self, CompressedImage, "/camera/camera/aligned_depth_to_color/image_raw/compressedDepth")
        ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], 10, 0.1)

        ts.registerCallback(self.callback)

        self.coord_pub = self.create_publisher(PointStamped, '/object_position', 10)

        # Ventanas para mostrar imágenes
        cv2.namedWindow("Detección RGB", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Detección RGB", 960, 720)
        cv2.namedWindow("Z Visual", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Z Visual", 640, 480)

        print("Nodo de detección 3D iniciado.")

        self._exit_requested = False

    def callback(self, rgb_msg, depth_msg):
        rgb_frame = self.decode_compressed_image(rgb_msg.data)
        depth_raw = self.decode_compressed_depth(depth_msg.data)
        if rgb_frame is None or depth_raw is None:
            return

        results = self.model.predict(rgb_frame, device=self.device, conf=0.4, verbose=False)
        annotated = results[0].plot() 

        depth_viz = cv2.convertScaleAbs(depth_raw, alpha=0.03)
        depth_viz = cv2.applyColorMap(depth_viz, cv2.COLORMAP_JET)


        print("RGB shape:", rgb_frame.shape)
        print("Depth shape:", depth_raw.shape)


        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx2d = int((x1 + x2) / 2)
            cy2d = int((y1 + y2) / 2)
            # Escalar coordenadas RGB (640x480) -> profundidad (480x270)
            # cx2d_depth = int(cx2d * (depth_raw.shape[1] / rgb_frame.shape[1]))
            # cy2d_depth = int(cy2d * (depth_raw.shape[0] / rgb_frame.shape[0]))


            if 0 <= cy2d < depth_raw.shape[0] and 0 <= cx2d < depth_raw.shape[1]:
                box_size = 9
                x_start = max(0, cx2d - box_size // 2)
                y_start = max(0, cy2d - box_size // 2)
                x_end = min(depth_raw.shape[1], cx2d + box_size // 2 + 1)
                y_end = min(depth_raw.shape[0], cy2d + box_size // 2 + 1)


                center_patch = depth_raw[y_start:y_end, x_start:x_end]
                valid_depths = center_patch[(center_patch > 100) & (center_patch < 10000)]

                if len(valid_depths) == 0:
                    print("Sin profundidad válida.")
                    continue

                depth_mm = np.median(valid_depths)
                # Z = depth_mm
                Z_raw = np.median(valid_depths)
                Z = self.a * Z_raw**2 + self.b * Z_raw + self.c  # profundidad corregida

                X_coord = (cx2d - self.cx) * Z / self.fx
                Y_coord = (cy2d - self.cy) * Z / self.fy
                # X_coord = (cx2d - self.cx) * Z / self.fx
                # Y_coord = -(cy2d - self.cy) * Z / self.fy
                # width_mm = (x2 - x1) * Z / self.fx

                width_px = x2 - x1
                # width_mm= (width_px * Z / self.fx)
                width_mm=50
                print(f"Detecciones: {len(results[0].boxes)}")

                print(f"Coordenadas: X = {X_coord:.1f} mm, Y = {Y_coord:.1f} mm, Z = {Z:.1f} mm")
                print(f"Ancho estimado: {width_mm:.1f} mm")


                msg = PointStamped()
                # msg.header.stamp = self.get_clock().now().to_msg()
                # msg.header.frame_id = "camera_link"
                msg.point.x = float(X_coord)
                msg.point.y = float(Y_coord)
                msg.point.z = float(Z)
                # msg.point.w=float(width_mm)
                self.coord_pub.publish(msg)

                # Dibujo en RGB
                # Clase y confianza
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = f"{self.model.names[cls_id]} {conf:.2f}"


                # Texto con fondo
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(annotated, (x1, y1 - text_h - 6), (x1 + text_w, y1), (255, 0, 0), -1)
                cv2.putText(annotated, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                # XYZ info
                cv2.putText(annotated, f"({X_coord:.1f},{Y_coord:.1f},{Z:.1f})mm", (x1, y2 + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(annotated, f"{width_mm:.1f} cm", (x1, y2 + 36),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

                # Dibujo en mapa de profundidad
                cv2.rectangle(depth_viz, (x_start, y_start), (x_end, y_end), (255, 255, 0), 2)
                cv2.putText(depth_viz, "Z centro", (x_start + 2, y_start - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                
                cv2.circle(annotated, (cx2d, cy2d), 5, (0, 255, 255), -1)
                cv2.putText(annotated, "Centro Z", (cx2d + 6, cy2d), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

                # Dibuja un círculo en la imagen de profundidad en la posición del centro de la detección
                cv2.circle(depth_viz, (cx2d, cy2d), 5, (0, 255, 255), -1)  # Amarillo
                cv2.putText(depth_viz, "Centro bbox", (cx2d + 6, cy2d), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)



        cv2.imshow("Detección RGB", annotated)
        cv2.imshow("Z Visual", depth_viz)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Finalizando desde ventana.")
            self._exit_requested = True

    def decode_compressed_image(self, data):
        np_arr = np.frombuffer(data, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    def decode_compressed_depth(self, data):
        try:
            byte_data = bytes(data)  # array.array a bytes
            start_idx = byte_data.find(b'\x89PNG')
            if start_idx == -1:
                print("Encabezado PNG no encontrado.")
                return None

            png_data = byte_data[start_idx:]
            np_arr = np.frombuffer(png_data, np.uint8)
            depth_img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)

            if depth_img is None:
                print("cv2.imdecode devolvió None.")
            return depth_img
        except Exception as e:
            print(f" Error al decodificar profundidad: {e}")
            return None

def main(args=None):
    rclpy.init(args=args)
    node = Object3DDetector()
    try:
        while rclpy.ok() and not node._exit_requested:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        print("Nodo terminado por el usuario.")
    finally:
        print("Cerrando nodo y ventanas...")
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
