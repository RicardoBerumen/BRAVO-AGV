#!/usr/bin/env python3
# Nodo ROS2 en Python que suscribe imágenes RGB y de profundidad y ejecuta detección 3D con YOLO

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
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

        # Buffer 
        self.rgb_frame = None
        self.depth_frame = None

        
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        self.get_logger().info(f'Torch CUDA disponible: {torch.cuda.is_available()}')
        self.model = YOLO('/home/evo/ros0_ws/src/bravo_agv/BRAVO_AGV/vision/Codigos/coca-cola-can-detection.yolov12/runs/train/EVO1/weights/best.pt')

        # Ventanas
        cv2.namedWindow('Detección 3D', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Detección 3D', 960, 720)
        cv2.namedWindow('Z visual', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Z visual', 640, 480)

        
        calib = np.load('/home/evo/ros0_ws/src/bravo_agv/BRAVO_AGV/vision/Codigos/coca-cola-can-detection.yolov12/camera_calibration2.npz')
        mtx, dist = calib['mtx'], calib['dist']
        self.fx, self.fy = mtx[0,0], mtx[1,1]
        self.cx, self.cy = mtx[0,2], mtx[1,2]

        
        self.create_timer(0.03, self.process_frames)

    def rgb_callback(self, msg):
        #OpenCV BGR
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        self.rgb_frame = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)

    def depth_callback(self, msg):
        # uint16 (mm)
        depth_mm = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
        self.depth_frame = np.array(depth_mm, dtype=np.uint16)

    def process_frames(self):
        if self.rgb_frame is None or self.depth_frame is None:
            return
        frame = self.rgb_frame.copy()
        depth = self.depth_frame.copy()

        
        results = self.model.predict(source=frame, device=self.device, conf=0.75, verbose=False)

        # profundidad
        depth_viz = cv2.convertScaleAbs(depth, alpha=0.03)
        depth_viz = cv2.applyColorMap(depth_viz, cv2.COLORMAP_JET)
        depth_viz = cv2.flip(depth_viz, 1)

        annotated = frame.copy()
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx2d = int((x1 + x2)/2)
                cy2d = int((y1 + y2)/2)
                box_size = 3
                half = box_size // 2
                x_start = max(0, cx2d - half)
                y_start = max(0, cy2d - half)
                x_end = min(depth.shape[1], cx2d + half + 1)
                y_end = min(depth.shape[0], cy2d + half + 1)
                center_patch = depth[y_start:y_end, x_start:x_end]
                valid = center_patch[(center_patch > 0) & (center_patch < 10000)]
                if valid.size == 0:
                    continue

                depth_mm = np.median(valid)
                Z = depth_mm/10.0  # convertir a cm
                X = (cx2d - self.cx)*Z/self.fx
                Y = -(cy2d - self.cy)*Z/self.fy
                # ancho
                # width_px = x2 - x1
                # width_cm = (width_px*Z/self.fx)
                width_cm=5
                ######################################################################
                cv2.rectangle(annotated, (x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(annotated, f"({X:.1f},{Y:.1f},{Z:.1f})cm", (x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),2)
                cv2.putText(annotated, f"{width_cm:.1f} cm", (x1,y2+15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255),2)
                cv2.rectangle(depth_viz, (x_start, y_start),(x_end,y_end),(255,255,0),2)
                cv2.putText(depth_viz, 'Z centro', (x_start+2, y_start-4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0),1)


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
