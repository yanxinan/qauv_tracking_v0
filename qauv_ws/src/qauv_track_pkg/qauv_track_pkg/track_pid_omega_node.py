#!/home/zb/anaconda3/envs/mujoco/bin/python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
import threading
import numpy as np
from std_msgs.msg import Float64
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Joy
from sensor_msgs.msg import Imu
from mypid import MyPID
from pyquaternion import Quaternion


"""
角速度闭环控制器
        l_y        r_y
        ^+         ^+
l_x <+   ->    <+   -> r_x
        v-         v-
r_x: 绕X轴旋转角速度, p, [-1, 1]
r_y: 绕Y轴旋转角速度, q, [-1, 1]
l_y: 沿Z轴平移, 油门距离控制
l_x: 绕Z轴旋转角速度, r, [-1, 1]
"""


class TrackPidOmegaNode(Node):
    def __init__(self, name):
        super().__init__(name)
        
        # 手柄与传感器数据
        self.l_x = 0.0
        self.l_y = 0.0
        self.l_z = 1.0
        self.r_x = 0.0
        self.r_y = 0.0
        self.r_z = 1.0
        self.imu_data = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.joy_lock = threading.Lock()
        self.imu_lock = threading.Lock()

        # 上一次推进器控制量
        self.last_pwm = [0.0, 0.0, 0.0, 0.0]

        # PID控制器
        # 角速度
        self.pid_p = MyPID(3, 0.4, 0.2, limit_out=10)
        self.pid_q = MyPID(3, 0.4, 0.2, limit_out=10)
        self.pid_r = MyPID(3, 0.4, 0.2, limit_out=10)
        
        # 创建订阅和发布
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.joy_sub = self.create_subscription(Joy, '/joy', self.joy_callback, qos)
        self.imu_sub = self.create_subscription(Imu, '/imu_data', self.imu_callback, qos)
        self.control_pub = self.create_publisher(Float64MultiArray, '/control_cmd', qos)
        self.error_pub = self.create_publisher(Float32MultiArray, '/error', qos)

        # 发布消息的格式与尺寸
        self.control_msg = Float64MultiArray()
        self.control_msg.data = [0.0] * 5

        # 创建定时器
        self.timer = self.create_timer(0.02, self.run)

        self.get_logger().info("Track Pid Omega Node Started")


    def joy_callback(self, msg: Joy):
        with self.joy_lock:
            self.l_x = -msg.axes[0]  # -1.0 ~ 1.0, 手柄的x轴正方向和习惯是相反的
            self.l_y = msg.axes[1]
            self.l_z = msg.axes[2]
            self.r_x = -msg.axes[3]
            self.r_y = msg.axes[4]
            self.r_z = msg.axes[5]


    def imu_callback(self, msg: Imu):
        with self.imu_lock:
            self.imu_data[0] = msg.orientation.x
            self.imu_data[1] = msg.orientation.y
            self.imu_data[2] = msg.orientation.z
            self.imu_data[3] = msg.orientation.w
            self.imu_data[4] = msg.angular_velocity.x
            self.imu_data[5] = msg.angular_velocity.y
            self.imu_data[6] = msg.angular_velocity.z
            self.imu_data[7] = msg.linear_acceleration.x
            self.imu_data[8] = msg.linear_acceleration.y
            self.imu_data[9] = msg.linear_acceleration.z


    def desire_wx_manual(self, r_x, r_y, l_x):
        # 期望角速度 [-1, 1]
        wx_d = r_x
        wy_d = r_y
        wz_d = l_x
        return wx_d, wy_d, wz_d


    def desire_throttle_manual(self, l_y):
        # 油门 [-5, 5]
        return (-l_y * 5)


    def close_omega_controller(self, wx_e, wy_e, wz_e, thro):
        # 角速度环
        tau_k = self.pid_p.pid_delta(wx_e)
        tau_m = self.pid_q.pid_delta(wy_e)
        tau_n = self.pid_r.pid_delta(wz_e)
        tau_z = thro
        # 控制增量
        pwm_m1_delta = (tau_k - tau_m + tau_z - tau_n) - self.last_pwm[0]
        pwm_m2_delta = (tau_k + tau_m + tau_z + tau_n) - self.last_pwm[1]
        pwm_m3_delta = (- tau_k + tau_m + tau_z - tau_n) - self.last_pwm[2]
        pwm_m4_delta = (- tau_k - tau_m + tau_z + tau_n) - self.last_pwm[3]
        pwm_m1_delta = min(max(pwm_m1_delta, -2), 2)
        pwm_m2_delta = min(max(pwm_m2_delta, -2), 2)
        pwm_m3_delta = min(max(pwm_m3_delta, -2), 2)
        pwm_m4_delta = min(max(pwm_m4_delta, -2), 2)
        # 控制量
        pwm_m1 = self.last_pwm[0] + pwm_m1_delta
        pwm_m2 = self.last_pwm[1] + pwm_m2_delta
        pwm_m3 = self.last_pwm[2] + pwm_m3_delta
        pwm_m4 = self.last_pwm[3] + pwm_m4_delta
        pwm_m1 = min(max(pwm_m1, -15), 15)
        pwm_m2 = min(max(pwm_m2, -15), 15)
        pwm_m3 = min(max(pwm_m3, -15), 15)
        pwm_m4 = min(max(pwm_m4, -15), 15)
        # 控制量存储
        self.last_pwm = [pwm_m1, pwm_m2, pwm_m3, pwm_m4]
        return pwm_m1, pwm_m2, pwm_m3, pwm_m4
    

    def calculate_force(self, f_1, f_2, f_3, f_4, q_c):
        tz_b = f_1 + f_2 + f_3 + f_4
        tx_w = 2 * (q_c.x*q_c.z + q_c.w*q_c.y) * tz_b
        ty_w = 2 * (q_c.y*q_c.z - q_c.w*q_c.x) * tz_b
        tz_w = (1 - 2 * (q_c.x*q_c.x + q_c.y*q_c.y)) * tz_b
        tk_b = 0.089 * (f_1 + f_2 - f_3 - f_4)
        tm_b = 0.089 * (-f_1 + f_2 + f_3 - f_4)
        tn_b = 1.2e-03 * (-f_1 + f_2 - f_3 + f_4)
        return tx_w, ty_w, tz_w, tk_b, tm_b, tn_b


    def run(self):
        with self.joy_lock:
            l_x, l_y, l_z, r_x, r_y, r_z,  = self.l_x, self.l_y, self.l_z, self.r_x, self.r_y, self.r_z
        with self.imu_lock:
            qx_c, qy_c, qz_c, qw_c, wx_c, wy_c, wz_c, ax_c, ay_c, az_c = self.imu_data
        
        # 角速度环
        wx_d, wy_d, wz_d = self.desire_wx_manual(r_x, r_y, l_x)
        wx_e = wx_d - wx_c
        wy_e = wy_d - wy_c
        wz_e = wz_d - wz_c
        thro = self.desire_throttle_manual(l_y)
        self.control_msg.data[0], self.control_msg.data[1], self.control_msg.data[2], self.control_msg.data[3] = self.close_omega_controller(wx_e, wy_e, wz_e, thro)
        self.control_msg.data[4]  = 0.5 if r_z < 0 else 0.0
        self.control_pub.publish(self.control_msg)
        # self.get_logger().info(f"{self.control_msg.data}")

        # 计算力和力矩
        # q_c = Quaternion(qw_c, qx_c, qy_c, qz_c)
        # tx_w, ty_w, tz_w, tk_b, tm_b, tn_b = self.calculate_force(self.control_msg.data[0], self.control_msg.data[1], self.control_msg.data[2], self.control_msg.data[3], q_c)

        error_msg = Float32MultiArray(data=[wx_d, wx_c, wy_d, wy_c, wz_d, wz_c])
        self.error_pub.publish(error_msg)


def main(args=None):
    rclpy.init(args=args)
    node = TrackPidOmegaNode("track_pid_omega_node")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().warn("KeyboardInterrupt, shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()
    