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
import torch
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common import policies
from my_normalization import ObsActNormalizer
import __main__


"""
位置闭环控制器
位置误差量程 [-320, 320], [-10, 250]

角度闭环控制器
        l_y        r_y
        ^+         ^+
l_x <+   ->    <+   -> r_x
        v-         v-
r_x: 绕X轴旋转角度, roll, [-pi, pi]
r_y: 绕Y轴旋转角度, pitch, [-pi, pi]
l_y: 沿Z轴平移, 油门距离控制
l_x: 绕Z轴旋转角度, yaw, 增量, [-1, 1]
"""


class CustomPolicy(policies.ActorCriticPolicy):
    """策略网络"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, net_arch=[32, 32, 32])

__main__.CustomPolicy = CustomPolicy

policy_path = "/home/zb/qauv_ws/src/qauv_track_pkg/qauv_track_pkg/bc_policy/bc_policy_finally.pt"
norm_path = "/home/zb/qauv_ws/src/qauv_track_pkg/qauv_track_pkg/bc_policy/norm_finally.npz"


class TrackBcPositionNode(Node):
    def __init__(self, name):
        super().__init__(name)

        # 期望的目标中心点坐标
        self.x_d = 320
        self.y_d = 240
        self.area_d = 1500
        
        # 手柄与传感器数据
        self.l_x = 0.0
        self.l_y = 0.0
        self.l_z = 1.0
        self.r_x = 0.0
        self.r_y = 0.0
        self.r_z = 1.0
        self.imu_data = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # qx, qy, qz, qw, wx, wy, wz, ax, ay, az
        self.img_data = [] # center_x, center_y, width, height, angle
        self.joy_lock = threading.Lock()
        self.imu_lock = threading.Lock()
        self.img_lock = threading.Lock()

        # 上一次期望四元数
        self.last_q_d = Quaternion(1, 0, 0, 0)

        # 上一次推进器控制量
        self.last_pwm = [0.0, 0.0, 0.0, 0.0]

        # bc策略 与 标准化参数
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy = torch.load(policy_path, weights_only=False, map_location=self.device)
        # print(self.policy)
        self.normalizer = ObsActNormalizer.load(norm_path)
        # print(self.normalizer)

        # PID控制器
        # 角度
        self.pid_roll = MyPID(2, 0.005, 0, limit_out=2)
        self.pid_pitch = MyPID(2, 0.005, 0, limit_out=2)
        self.pid_yaw = MyPID(2, 0, 0, limit_out=2)
        # 角速度
        self.pid_p = MyPID(3, 0.4, 0.2, limit_out=10)
        self.pid_q = MyPID(3, 0.4, 0.2, limit_out=10)
        self.pid_r = MyPID(3, 0.4, 0.2, limit_out=10)
        self.pid_thro = MyPID(0.002, 0.00001, 0.0001, limit_out=10)

        # 创建订阅和发布
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.joy_sub = self.create_subscription(Joy, '/joy', self.joy_callback, qos)
        self.imu_sub = self.create_subscription(Imu, '/imu_data', self.imu_callback, qos)
        self.img_sub = self.create_subscription(Float64MultiArray, '/img_data', self.img_callback, qos)
        self.control_pub = self.create_publisher(Float64MultiArray, '/control_cmd', qos)
        self.error_pub = self.create_publisher(Float32MultiArray, '/error', qos)

        # 发布消息的格式与尺寸
        self.control_msg = Float64MultiArray()
        self.control_msg.data = [0.0] * 5

        # 创建定时器
        self.timer = self.create_timer(0.02, self.run)

        self.get_logger().info("Track Bc Position Node Started")


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


    def img_callback(self, msg: Float64MultiArray):
        with self.img_lock:
            self.img_data = msg.data


    def desire_quat_manual(self, r_x, r_y, yaw_c):
        # 机体坐标系下的期望旋转
        q_roll = Quaternion(axis=[1, 0, 0], angle=r_x * np.pi)
        q_pitch = Quaternion(axis=[0, 1, 0], angle=r_y * np.pi)
        # 机体坐标系下的期望姿态, 注意顺序: 先roll后pitch
        q_body = q_pitch * q_roll
        # 转换到世界坐标系
        q_yaw = Quaternion(axis=[0, 0, 1], angle=yaw_c)
        q_world = q_yaw * q_body * q_yaw.inverse
        return q_world


    def desire_throttle_manual(self, l_y):
        # 油门 [-5, 5]
        return (-l_y * 5)
    

    def desire_area_manual(self, l_y):
        # 面积 [1000, 9000]
        return (l_y * 4000 + 5000)


    def error_quat(self, q_d: Quaternion, q_c: Quaternion):
        # 倾转分离, 倾斜误差四元数
        # 提取Z轴向量
        z_c = q_c.rotate([0, 0, 1]) # 单位四元数, Z轴向量模长为1
        z_d = q_d.rotate([0, 0, 1]) # 单位四元数, Z轴向量模长为1
        # 计算从z_c到z_d的"最短弧旋转"四元数
        cross = np.cross(z_c, z_d) # 叉乘 |cross| = |z_c||z_d|sin(theta) = sin(theta)
        cross_norm = np.linalg.norm(cross)
        dot = np.dot(z_c, z_d) # 点乘 dot = |z_c||z_d|cos(theta) = cos(theta)
        if cross_norm < 1e-5:
            # 夹角为0或180度
            q_d_red = q_d # 直接取q_d(避免退化)
        else:
            # 构造旋转四元数(Rodrigues公式)
            angle = np.arctan2(cross_norm, dot) # 夹角
            axis = cross / cross_norm # 归一化轴
            q_rot = Quaternion(axis=axis, angle=angle)
            q_d_red = q_rot * q_c
        # 倾转误差四元数
        q_e_tilt = q_c.inverse * q_d_red
        # pitch=90°, 抑制roll抖动
        ortho_thr = 0.15 # acos(0.05)=87.13°, acos(0.1)=84.26°, acos(0.2)=78.46°
        almost_90 = abs(dot) < ortho_thr
        if almost_90:
            # q_e_tilt = Quaternion(q_e_tilt.w, 0.0, q_e_tilt.y, q_e_tilt.z) # 硬死区
            q_e_tilt = Quaternion(q_e_tilt.w, q_e_tilt.x * abs(dot) / ortho_thr, q_e_tilt.y, q_e_tilt.z) # 软死区
        return q_e_tilt


    def error_yaw_manual(self, l_x):
        # 自旋误差弧度 [-1, 1]
        return l_x
    

    def close_position_controller_delta(self, x_e, y_e, a_c, yaw_c, pitch_c, roll_c):
        # 位置环
        obs = np.array([x_e, y_e, a_c])
        obs_norm = self.normalizer.normalize_obs(obs).astype(np.float32)
        input_tensor = torch.from_numpy(obs_norm).unsqueeze(0).to(self.device)
        with torch.no_grad():
            act_norm = self.policy(input_tensor)[0].squeeze().cpu().numpy()
        acts = self.normalizer.denormalize_act(act_norm)
        roll_d = roll_c + 5.0*acts[0]
        pitch_d = pitch_c + 5.0*acts[1]
        # 机体坐标系下的期望旋转
        q_roll = Quaternion(axis=[1, 0, 0], angle=roll_d)
        q_pitch = Quaternion(axis=[0, 1, 0], angle=pitch_d)
        # 机体坐标系下的期望姿态, 习惯顺序: 先roll后pitch
        q_body = q_pitch * q_roll
        # 转换到世界坐标系
        q_yaw = Quaternion(axis=[0, 0, 1], angle=yaw_c)
        q_world = q_yaw * q_body * q_yaw.inverse
        return q_world, acts[0], acts[1]


    def close_angle_controller(self, q_e: Quaternion, yaw_e):
        # 角度环
        wx_d = self.pid_roll.pid_delta(q_e.x)
        wy_d = self.pid_pitch.pid_delta(q_e.y)
        wz_d = self.pid_yaw.pid_delta(yaw_e)
        return wx_d, wy_d, wz_d


    def close_omega_controller_manual(self, wx_e, wy_e, wz_e, thro):
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
    

    def close_omega_controller_auto(self, wx_e, wy_e, wz_e, area_e):
        # 角速度环
        tau_k = self.pid_p.pid_delta(wx_e)
        tau_m = self.pid_q.pid_delta(wy_e)
        tau_n = self.pid_r.pid_delta(wz_e)
        tau_z = self.pid_thro.pid_delta(-area_e)
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


    def run(self):
        with self.joy_lock:
            l_x, l_y, l_z, r_x, r_y, r_z,  = self.l_x, self.l_y, self.l_z, self.r_x, self.r_y, self.r_z
        with self.imu_lock:
            qx_c, qy_c, qz_c, qw_c, wx_c, wy_c, wz_c, ax_c, ay_c, az_c = self.imu_data
        with self.img_lock:
            if not self.img_data:
                x_c, y_c, width, height, angle = None, None, None, None, None
            else:
                x_c, y_c, width, height, angle = self.img_data
                # self.get_logger().info(f"area = {width * height}")

        yaw_c = np.arctan2(2.0*(qw_c*qz_c+qx_c*qy_c), 1.0-2.0*(qy_c*qy_c+qz_c*qz_c))
        pitch_c = np.arcsin(np.clip(2*(qw_c*qy_c-qz_c*qx_c), -1.0, 1.0))
        roll_c = np.arctan2(2.0*(qw_c*qx_c+qy_c*qz_c), 1.0-2.0*(qx_c*qx_c+qy_c*qy_c))
        if x_c is not None and y_c is not None:
            # 位置环
            x_e = self.x_d - x_c
            y_e = self.y_d - y_c
            a_c = width * height
            q_d, delta_roll_d, delta_pitch_d = self.close_position_controller_delta(x_e, y_e, a_c, yaw_c, pitch_c, roll_c)
            # 期望四元数, 球面线性插值(SLERP)
            q_d = Quaternion.slerp(self.last_q_d, q_d, 0.15)  # α=0.1~0.2 可调
            self.last_q_d = q_d
            # 角度环
            q_c = Quaternion(qw_c, qx_c, qy_c, qz_c)
            q_e = self.error_quat(q_d, q_c)
            yaw_e = self.error_yaw_manual(l_x)
            wx_d, wy_d, wz_d = self.close_angle_controller(q_e, yaw_e)
            # 角速度环
            wx_e = wx_d - wx_c
            wy_e = wy_d - wy_c
            wz_e = wz_d - wz_c
            # self.area_d = self.desire_area_manual(l_y)
            area_e = self.area_d - a_c
            self.control_msg.data[0], self.control_msg.data[1], self.control_msg.data[2], self.control_msg.data[3] = self.close_omega_controller_auto(wx_e, wy_e, wz_e, area_e)
            self.control_msg.data[4]  = 0.5 if r_z < 0 else 0.0
            self.control_pub.publish(self.control_msg)
            # self.get_logger().info(f"{self.control_msg.data}")
        else:
            x_e, y_e, area_e, a_c = 0, 0, 0, 0
            delta_roll_d, delta_pitch_d = 0, 0
            q_d = self.desire_quat_manual(r_x, r_y, yaw_c)
            # 期望四元数, 球面线性插值(SLERP)
            q_d = Quaternion.slerp(self.last_q_d, q_d, 0.15)  # α=0.1~0.2 可调
            self.last_q_d = q_d
            # 角度环
            q_c = Quaternion(qw_c, qx_c, qy_c, qz_c)
            q_e = self.error_quat(q_d, q_c)
            yaw_e = self.error_yaw_manual(l_x)
            wx_d, wy_d, wz_d = self.close_angle_controller(q_e, yaw_e)
            # 角速度环
            wx_e = wx_d - wx_c
            wy_e = wy_d - wy_c
            wz_e = wz_d - wz_c
            thro = self.desire_throttle_manual(l_y)
            self.control_msg.data[0], self.control_msg.data[1], self.control_msg.data[2], self.control_msg.data[3] = self.close_omega_controller_manual(wx_e, wy_e, wz_e, thro)
            self.control_msg.data[4]  = 0.5 if r_z < 0 else 0.0
            self.control_pub.publish(self.control_msg)
            # self.get_logger().info(f"{self.control_msg.data}")

        error_msg = Float32MultiArray(data=[x_e, y_e, area_e, a_c, delta_roll_d, delta_pitch_d, q_e.x, q_e.y])
        self.error_pub.publish(error_msg)


def main(args=None):
    rclpy.init(args=args)
    node = TrackBcPositionNode("track_bc_position_node")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().warn("KeyboardInterrupt, shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()
    