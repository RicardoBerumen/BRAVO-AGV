#!/usr/bin/env python3

#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32
import numpy as np
import cv2
import torch
from ultralytics import YOLO

class Solo_RGB(Node):
    def __init__(self):
        super().__init__('rgb_only_detection_node')

        from rclpy.qos import qos_profile_sensor_data
        self.create_subscription(CompressedImage, '/out/compressed', self.rgb_callback, qos_profile_sensor_data)

        self.x_pub = self.create_publisher(Float32, 'object/center_x', 10)
        self.y_pub = self.create_publisher(Float32, 'object/center_y', 10)

        self.rgb_frame = None

        self.device = 0 if torch.cuda.is_available() else 'cpu'
        self.get_logger().info(f'Torch CUDA disponible: {torch.cuda.is_available()}')
        self.model = YOLO('/home/evo/ros0_ws/src/bravo_agv/BRAVO_AGV/vision/Codigos/coca-cola-can-detection.yolov12/runs/train/EVO1/weights/best.pt')

        cv2.namedWindow('Detección RGB', cv2.WINDOW_NORMAL)
        self.create_timer(1.0 / 30.0, self.process_frame)

    def rgb_callback(self, msg: CompressedImage):
        np_arr = np.frombuffer(msg.data, np.uint8)
        self.rgb_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # ya en BGR

    def process_frame(self):
        if self.rgb_frame is None:
            return

        frame = self.rgb_frame.copy()
        results = self.model.predict(source=frame, device=self.device, conf=0.75, verbose=False)
        annotated = frame.copy()

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx2d = int((x1 + x2) / 2)
                cy2d = int((y1 + y2) / 2)

                self.x_pub.publish(Float32(data=cx2d))
                self.y_pub.publish(Float32(data=cy2d))

                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated, f"Center: ({cx2d}, {cy2d})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        cv2.imshow('Detección RGB', annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.get_logger().info('Finalizando...')
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = Solo_RGB()
    rclpy.spin(node)
    node.destroy_node()

if __name__ == '__main__':
    main()
