#!/usr/bin/env python3
# Nodo ROS2 en Python para publicar imágenes RGB y de profundidad desde una Orbbec Astra Pro Plus usando OpenNI2

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
from primesense import openni2

class AstraPublisher(Node):
    def __init__(self):
        super().__init__('astra_openni_publisher')
        self.bridge = CvBridge()
        self.rgb_pub = self.create_publisher(Image, 'camera/rgb/image_raw', 10)
        self.depth_pub = self.create_publisher(Image, 'camera/depth/image_raw', 10)


        openni2_path = (
            "/home/evo/Documents/orbbec/Orbbec_descargas/Orbbec_OpenNI_v2.3.0.86-beta6_linux_release/"
            "OpenNI_2.3.0.86_202210111154_4c8f5aa4_beta6_linux_x64/"
            "OpenNI_2.3.0.86_202210111154_4c8f5aa4_beta6_linux/sdk/libs/"
        )
        openni2.initialize(openni2_path)
        self.dev = openni2.Device.open_any()

        #RGB 1280x960, profundidad 640x480
        self.color_stream = self.dev.create_color_stream()
        self.color_stream.set_video_mode(
            openni2.OniVideoMode(
                pixelFormat=openni2.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888,
                resolutionX=1280, resolutionY=960, fps=30
            )
        )
        self.depth_stream = self.dev.create_depth_stream()
        self.depth_stream.set_video_mode(
            openni2.OniVideoMode(
                pixelFormat=openni2.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM,
                resolutionX=640, resolutionY=480, fps=30
            )
        )
        self.color_stream.start()
        self.depth_stream.start()

        # 30 Hz
        self.timer = self.create_timer(1.0/30.0, self.timer_callback)

    def timer_callback(self):
        #frame RGB
        frame_color = self.color_stream.read_frame()
        data_color = np.frombuffer(frame_color.get_buffer_as_triplet_uint8(), dtype=np.uint8)
        color_image = data_color.reshape((frame_color.height, frame_color.width, 3))

        # frame de profundidad
        frame_depth = self.depth_stream.read_frame()
        data_depth = np.frombuffer(frame_depth.get_buffer_as_uint16(), dtype=np.uint16)
        depth_image = data_depth.reshape((frame_depth.height, frame_depth.width))

        #ROS
        timestamp = self.get_clock().now().to_msg()
        msg_rgb = self.bridge.cv2_to_imgmsg(color_image, encoding='rgb8')
        msg_rgb.header.stamp = timestamp
        msg_rgb.header.frame_id = 'camera'

        # Profundidad 
        msg_depth = self.bridge.cv2_to_imgmsg(depth_image, encoding='16UC1')
        msg_depth.header.stamp = timestamp
        msg_depth.header.frame_id = 'camera'


        self.rgb_pub.publish(msg_rgb)
        self.depth_pub.publish(msg_depth)

    def destroy(self):
        self.color_stream.stop()
        self.depth_stream.stop()
        openni2.unload()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = AstraPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
