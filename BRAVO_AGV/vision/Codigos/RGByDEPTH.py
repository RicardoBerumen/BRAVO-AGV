import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2
from primesense import openni2


class AstraHybridPublisher(Node):
    def __init__(self):
        super().__init__('astra_hybrid_publisher')
        self.bridge = CvBridge()

        self.rgb_pub = self.create_publisher(Image, 'camera/rgb/image_raw', 10)
        self.depth_pub = self.create_publisher(Image, 'camera/depth/image_raw', 10)

        # === RGB OpenCV (/dev/video1) ===
        self.cap = cv2.VideoCapture(1)
        if not self.cap.isOpened():
            self.get_logger().error("No /dev/video1")
            exit(1)


        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
        self.cap.set(cv2.CAP_PROP_FPS, 30)


        openni2_path = (
            "/home/evo/Documents/orbbec/Orbbec_descargas/Orbbec_OpenNI_v2.3.0.86-beta6_linux_release/"
            "OpenNI_2.3.0.86_202210111154_4c8f5aa4_beta6_linux_x64/"
            "OpenNI_2.3.0.86_202210111154_4c8f5aa4_beta6_linux/sdk/libs/"
        )
        openni2.initialize(openni2_path)
        self.dev = openni2.Device.open_any()

        self.depth_stream = self.dev.create_depth_stream()
        self.depth_stream.set_video_mode(
            openni2.OniVideoMode(
                pixelFormat=openni2.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM,
                resolutionX=640, resolutionY=480, fps=30
            )
        )
        self.depth_stream.start()

        self.timer = self.create_timer(1.0 / 30.0, self.timer_callback)

    def timer_callback(self):
        timestamp = self.get_clock().now().to_msg()

        # === RGB ===
        ret, frame = self.cap.read()
        if ret:
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            msg_rgb = self.bridge.cv2_to_imgmsg(rgb_image, encoding='rgb8')
            msg_rgb.header.stamp = timestamp
            msg_rgb.header.frame_id = 'camera_rgb_optical_frame'
            self.rgb_pub.publish(msg_rgb)
        else:
            self.get_logger().warn("No frame RGB")

        # === Depth ===
        frame_depth = self.depth_stream.read_frame()
        data_depth = np.frombuffer(frame_depth.get_buffer_as_uint16(), dtype=np.uint16)
        depth_image = data_depth.reshape((frame_depth.height, frame_depth.width))
        depth_image = np.ascontiguousarray(depth_image)

        msg_depth = self.bridge.cv2_to_imgmsg(depth_image, encoding='8UC1')
        msg_depth.header.stamp = timestamp
        msg_depth.header.frame_id = 'camera_depth_optical_frame'
        self.depth_pub.publish(msg_depth)

    def destroy(self):
        self.cap.release()
        self.depth_stream.stop()
        openni2.unload()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = AstraHybridPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
