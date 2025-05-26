#!/usr/bin/env python3
# Nodo ROS2 en Python que suscribe imágenes RGB y de profundidad, ejecuta detección 3D con YOLO y publica XYZ y ancho

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from cv_bridge import CvBridge
import numpy as np
import cv2
import torch
from ultralytics import YOLO

class AstraDetection(Node):
    def __init__(self):
        super().__init__('astra_detection_node')
        self.bridge = CvBridge()

        self.create_subscription(Image, 'camera/rgb/image_raw', self.rgb_callback, 10)
        self.create_subscription(Image, 'camera/depth/image_raw', self.depth_callback, 10)


        self.x_pub = self.create_publisher(Float32, 'object/x', 10)
        self.y_pub = self.create_publisher(Float32, 'object/y', 10)
        self.z_pub = self.create_publisher(Float32, 'object/z', 10)
        self.width_pub = self.create_publisher(Float32, 'object/width', 10)


        self.rgb_frame = None
        self.depth_frame = None

        self.device = 0 if torch.cuda.is_available() else 'cpu'
        self.get_logger().info(f'Torch CUDA disponible: {torch.cuda.is_available()}')
        self.model = YOLO('/home/evo/ros0_ws/src/bravo_agv/BRAVO_AGV/vision/Codigos/coca-cola-can-detection.yolov12/runs/train/EVO1/weights/best.pt')

        cv2.namedWindow('Detección 3D', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Detección 3D', 960, 720)
        cv2.namedWindow('Z visual', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Z visual', 640, 480)


        calib = np.load('/home/evo/ros0_ws/src/bravo_agv/BRAVO_AGV/vision/Codigos/coca-cola-can-detection.yolov12/camera_calibration2.npz')
        mtx = calib['mtx']
        self.fx, self.fy = mtx[0, 0], mtx[1, 1]
        self.cx, self.cy = mtx[0, 2], mtx[1, 2]

        # timer a ~30 Hz
        self.create_timer(1.0/30.0, self.process_frames)

    def rgb_callback(self, msg):
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        self.rgb_frame = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)

    def depth_callback(self, msg):
        depth_mm = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
        self.depth_frame = np.array(depth_mm, dtype=np.uint16)

    def process_frames(self):
        if self.rgb_frame is None or self.depth_frame is None:
            return
        frame = self.rgb_frame.copy()
        depth = self.depth_frame.copy()

        results = self.model.predict(source=frame, device=self.device, conf=0.75, verbose=False)

        depth_viz = cv2.convertScaleAbs(depth, alpha=0.03)
        depth_viz = cv2.applyColorMap(depth_viz, cv2.COLORMAP_JET)
        depth_viz = cv2.flip(depth_viz, 1)

        annotated = frame.copy()
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx2d = int((x1 + x2) / 2)
                cy2d = int((y1 + y2) / 2)


                box_size = 4
                half = box_size // 2
                xs = max(0, cx2d - half)
                ys = max(0, cy2d - half)
                xe = min(depth.shape[1], cx2d + half + 1)
                ye = min(depth.shape[0], cy2d + half + 1)
                patch = depth[ys:ye, xs:xe]
                valid = patch[(patch > 0) & (patch < 10000)]
                if valid.size == 0:
                    continue

                depth_mm = np.median(valid)
                Z = depth_mm / 10.0  # cm
                X = (cx2d - self.cx) * Z / self.fx
                Y = -(cy2d - self.cy) * Z / self.fy
                # ancho
                # width_px = x2 - x1
                # width_cm = (width_px*Z/self.fx)
                width_cm=5 #Medida lata coca-cola

                self.x_pub.publish(Float32(data=X))
                self.y_pub.publish(Float32(data=Y))
                self.z_pub.publish(Float32(data=Z))
                self.width_pub.publish(Float32(data=width_cm))

                # imagen RGB
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated, f"({X:.1f},{Y:.1f},{Z:.1f})cm", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(annotated, f"{width_cm:.1f} cm", (x1, y2 + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

                # profundidad
                cv2.rectangle(depth_viz, (xs, ys), (xe, ye), (255, 255, 0), 2)
                cv2.putText(depth_viz, 'Z centro', (xs + 2, ys - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        cv2.imshow('Detección 3D', annotated)
        cv2.imshow('Z visual', depth_viz)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.get_logger().info('Finalizando...')
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = AstraDetection()
    rclpy.spin(node)
    node.destroy_node()

if __name__ == '__main__':
    main()
