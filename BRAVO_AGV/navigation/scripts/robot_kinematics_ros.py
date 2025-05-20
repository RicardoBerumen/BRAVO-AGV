import serial
from math import pi
from time import sleep

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from geometry_msgs.msg import Twist

class robot_control(Node):
    """
    A node to read the velocity topic and write the necessary uart commands
    """
    def __init__(self):
        super().__init__("robot_control")
        self.uart_config()
        self.vel_subscription = self.create_subscription(
            Twist,
            'cmd_vel',
            self.vel_callback,
            10
        )
        self.vel_subscription
        """
        Robot Kinematics Definition
        """
        

        self.l = 0.5
        self.w_r = 0.1016
    
    def uart_config(self):
        """
        Config section
        """
        self.ser = serial.Serial("/dev/ttyAMA0", 115200)
        ser = self.ser
        print("Configuration ready")
        print(ser.name)
        self.newline = "\n"

    def vel_callback(self, msg):
        self.get_logger().info("I heard: %d", %msg.linear.x)
        self.V = msg.linear.x
        self.w = msg.angular.z

        V = self.V
        w = self.w
        l = self.l
        w_r = self.w_r

        vr = (2*V + l*w)/2
        vl = (2*V - l*w)/2

        vr_rpm = rpm(vr, w_r)
        vl_rpm = rpm(vl, w_r)

        self.vr_rpm = vr_rpm
        self.vl_rpm = vl_rpm

        """
        UART Encoding and communication
        """
        # Encoding values
        vrn = str(vr_rpm)
        vln = str(vl_rpm)
        print(vrn)
        print(vln)

        i = 0
        ser = self.ser 
        newline = self.newline
        try:
            ser.write(b'1 \n')
            ser.write((vrn.encode()+newline.encode()))
            ser.write(b'2 \n')
            ser.write((vln.encode()+newline.encode()))
            sleep(0.01)

        except KeyboardInterrupt:
            print("closing")
        ser.close()



def rpm(ms, w):
    v_rpm = ms/w*2*pi/60
    return round(v_rpm, 6)

def main(args = None):

    """
    ROS Node
    """
    rclpy.init(args=args)
    velocity_subscriber = robot_control()
    rclpy.spin(velocity_subscriber)
    
    # destroy node
    velocity_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()