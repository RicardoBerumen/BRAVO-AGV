import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
import numpy as np
import cv2
import pyrealsense2 as rs
import torch
from ultralytics import YOLO

class Object3DDetector(Node):
    def __init__(self):
        super().__init__('object_3d_detector')


        with np.load("/home/evo/Documents/orbbec/Orbbec_descargas/VISION-BRAVO-main/Codigos/camera_calibration_logi.npz") as X:
            self.mtx, self.dist = X['mtx'], X['dist']
            self.fx, self.fy = self.mtx[0, 0], self.mtx[1, 1]
            self.cx, self.cy = self.mtx[0, 2], self.mtx[1, 2]


        depth_model = np.load("/home/evo/Documents/orbbec/Orbbec_descargas/VISION-BRAVO-main/Codigos/depth_model_fit_yolo.npz")
        self.a, self.b, self.c = depth_model["a"], depth_model["b"], depth_model["c"]


        self.model = YOLO("/home/evo/Documents/coca-cola-can-detection.yolov12/runs/train/EVO1/weights/best.pt")
        self.device = 0 if torch.cuda.is_available() else 'cpu'


        self.coord_pub = self.create_publisher(PointStamped, '/object_position', 10)

        # Configuración de la RealSense
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)


        cv2.namedWindow("Detección RGB", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Detección RGB", 960, 720)
        cv2.namedWindow("Profundidad", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Profundidad", 640, 480)

        self._exit_requested = False
        print("✅ Nodo de detección 3D iniciado con Intel RealSense.")

    def process_frame(self):
        deteccion_hecha = False

        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        aligned_frames = self.align.process(frames)
        
        if not depth_frame or not color_frame:
            return



        depth_raw = np.asanyarray(depth_frame.get_data())
        rgb_frame = np.asanyarray(color_frame.get_data())


        results = self.model.predict(rgb_frame, device=self.device, conf=0.4, verbose=False)
        annotated = results[0].plot()
        depth_viz = cv2.convertScaleAbs(depth_raw, alpha=0.03)
        depth_viz = cv2.applyColorMap(depth_viz, cv2.COLORMAP_JET)

        for box in results[0].boxes:
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx2d = int((x1 + x2) / 2)
            cy2d = int((y1 + y2) / 2)

            if 0 <= cy2d < depth_raw.shape[0] and 0 <= cx2d < depth_raw.shape[1]:
                box_size = 9
                x_start = max(0, cx2d - box_size // 2)
                y_start = max(0, cy2d - box_size // 2)
                x_end = min(depth_raw.shape[1], cx2d + box_size // 2 + 1)
                y_end = min(depth_raw.shape[0], cy2d + box_size // 2 + 1)

                patch = depth_raw[y_start:y_end, x_start:x_end]
                valid_depths = patch[(patch > 0) & (patch < 10000)]

                if len(valid_depths) == 0:
                    continue

                Z_raw = np.median(valid_depths)
                Z = self.a * Z_raw**2 + self.b * Z_raw + self.c

                X_coord = (cx2d - self.cx) * Z / self.fx
                Y_coord = (cy2d - self.cy) * Z / self.fy

                msg = PointStamped()
                msg.point.x = float(X_coord)
                msg.point.y = float(Y_coord)
                msg.point.z = float(Z)
                self.coord_pub.publish(msg)
                deteccion_hecha = True



                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = f"{self.model.names[cls_id]} {conf:.2f}"

                # cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # cv2.putText(annotated, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                # cv2.circle(annotated, (cx2d, cy2d), 5, (255, 255, 0), -1)

                # cv2.rectangle(depth_viz, (x_start, y_start), (x_end, y_end), (255, 255, 0), 2)
                # cv2.circle(depth_viz, (cx2d, cy2d), 5, (255, 255, 0), -1)

                                # Dibujo en mapa de profundidad
                cv2.rectangle(depth_viz, (x_start, y_start), (x_end, y_end), (255, 255, 0), 2)
                cv2.putText(depth_viz, "Z centro", (x_start + 2, y_start - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                
                cv2.circle(annotated, (cx2d, cy2d), 5, (0, 255, 255), -1)
                cv2.putText(annotated, "Centro Z", (cx2d + 6, cy2d), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

                


        if not deteccion_hecha:
            msg = PointStamped()
            msg.point.x = 999999999.0
            msg.point.y = 999999999.0
            msg.point.z = 999999999.0
            self.coord_pub.publish(msg)


        cv2.imshow("Detección RGB", annotated)
        cv2.imshow("Profundidad", depth_viz)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self._exit_requested = True

    def destroy(self):
        self.pipeline.stop()
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = Object3DDetector()
    try:
        while rclpy.ok() and not node._exit_requested:
            node.process_frame()
            rclpy.spin_once(node, timeout_sec=0.01)
    except KeyboardInterrupt:
        print("Terminación por teclado.")
    finally:
        print("Cerrando nodo y pipeline.")
        node.destroy()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
