import sys
import serial
from math import pi, sqrt, atan2, tanh, cos, sin
from time import sleep

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile
from geometry_msgs.msg import PointStamped, Twist

class obj_control(Node):
    """
    A node to read the velocity topic and write the necessary uart commands
    """
    def __init__(self):
        super().__init__("obj_control")
        self.obj_subscription = self.create_subscription(
            PointStamped,
            '/object_position',
            self.obj_callback,
            10
        )
        self.obj_subscription
        qos = QoSProfile(depth=10)
        self.vel_pub = self.create_publisher(Twist, '/cmd_vel', qos)
        """
        Robot Kinematics Definition
        """
        self.l = 0.63
        self.w_r = 0.1016
        self.kpr = 1.0
        self.kpt = 1.0
        self.kgamma = 1.0
        self.v_max = 3.0

        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.z_d = 0.0
        self.x_d = 0.0
        self.y_d = 0.0
        self.x_df = 0.0
        self.y_df = 0.0
        self.get_clock().now()
        self.tiempo = self.get_clock().now().nanoseconds/1000000000
        self.tiempo_ant = self.tiempo

    def obj_callback(self, msg):
        if msg.point.z < 99999:
            self.z_d = msg.point.z
            self.x_d = msg.point.x
            self.y_d = msg.point.y
            self.x_df = self.x + self.z_d/1000.0
            self.y_df = self.y + self.x_d/1000.0
        self.get_logger().info('Z distance: "{0}"'.format(msg.point.z/1000))
        self.get_logger().info('X distance: "{0}"'.format(msg.point.x/1000))
        self.get_logger().info('Y distance: "{0}"'.format(msg.point.y/1000))
        self.tiempo = self.get_clock().now().nanoseconds/1000000000
        x_df = self.x_df
        y_df = self.y_df

        try:
            thetad = atan2((y_df-self.y), (x_df-self.x))
            thetae = self.theta - thetad

            self.d = x_df - self.x
            self.get_logger().info('Position X: "{0}"'.format(self.x))
            self.get_logger().info('Distance to target: "{0}"'.format(self.d))
            self.gamma = -tanh((self.kgamma)*abs(thetae))+1

            if self.d > self.l:
                self.V = self.gamma*self.v_max*tanh((self.kpt*self.d**2)/self.v_max)
            else:
                self.V = 0.0

            twist = Twist()
            twist.linear.x = self.V
            twist.linear.y = 0.0
            twist.linear.z = 0.0

            twist.angular.x = 0.0
            twist.angular.y = 0.0
            twist.angular.z = 0.0
            self.vel_pub.publish(twist)

            self.dt = self.tiempo - self.tiempo_ant
            self.tiempo_ant = self.tiempo
            self.x = self.x + self.V*self.dt


        except KeyboardInterrupt or ExternalShutdownException:
            print("closing")


def main(args = None):

    """
    ROS Node
    """
    rclpy.init(args=args)
    try:
        obj_subscriber = obj_control()
        rclpy.spin(obj_subscriber)
    except KeyboardInterrupt:
        pass
    except ExternalShutdownException:
        sys.exit(1)
    finally:
        # destroy node
        rclpy.try_shutdown()
        obj_subscriber.destroy_node()
        
if __name__ == '__main__':
    main()