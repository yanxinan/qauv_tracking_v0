#!/home/zb/anaconda3/envs/mujoco/bin/python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
import time
import threading
import mujoco
import mujoco.viewer
import cv2
import glfw
import numpy as np
from std_msgs.msg import Float64
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Imu
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class MuJoCoSimulatorNode(Node):
    def __init__(self, name):
        super().__init__(name)
        
        # 主线程运行标志
        self._running = True
        
        # 控制数据
        self.control_data = [0.0] * 6
        self.control_lock = threading.Lock()

        # 查看器就绪标志
        self.viewer_ready = threading.Event()

        # 创建订阅和发布
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.control_sub = self.create_subscription(Float64MultiArray, '/control_cmd', self.control_callback, qos)
        self.imu_pub = self.create_publisher(Imu, '/imu_data', qos)
        self.dep_pub = self.create_publisher(Float64, '/dep_data', qos)

        # 发布消息的格式与尺寸
        self.imu_msg = Imu()
        self.dep_msg = Float64()
        
        # OpenCV/ROS图像转换
        self.bridge = CvBridge()

        # 设置发布频率
        self.pub_sensor_rate = 50.0
        self.pub_sensor_interval = 1.0 / self.pub_sensor_rate
        self.last_pub_sensor_time = time.time()

        # 创建仿真环境
        self.model = mujoco.MjModel.from_xml_path('/home/zb/qauv_ws/src/mujoco_pkg/mujoco_pkg/qauv_track.xml')
        self.data = mujoco.MjData(self.model)

        # 创建仿真线程
        self.sim_thread = threading.Thread(target=self.simulator)
        self.sim_thread.daemon = True

        # 初始化查看器
        try:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer_ready.set()
            self.get_logger().info("Viewer initialized")
        except Exception as e:
            self.get_logger().error(f"Viewer init failed: {str(e)}")
            self.viewer = None

        # 图像相关
        self.window = None
        self.camera = None
        self.scene = None
        self.mjr_context = None

        # 启动仿真线程
        self.sim_thread.start()
        
        self.get_logger().info("MuJoCo Simulator Node Started")


    def destroy_node(self):
        self._running = False

        # 关闭仿真线程
        if self.sim_thread.is_alive():
            self.sim_thread.join(timeout=1.0)

        # 关闭查看器
        if self.viewer is not None:
            self.viewer.close()

        # 关闭 OpenCV 窗口
        cv2.destroyAllWindows()

        # 处理 GLFW 窗口
        if hasattr(self, "window") and self.window:
            glfw.destroy_window(self.window)
            glfw.terminate()

        # 调用 ROS2 的销毁操作
        super().destroy_node()


    def control_callback(self, msg):
        with self.control_lock:
            self.control_data[:5] = msg.data

    
    def get_sensor_data(self):
        acce = self.data.sensor("body_acce").data
        gyro = self.data.sensor("body_gyro").data
        quat = self.data.sensor("body_quat").data
        dep = self.data.sensor("body_dep").data[2]
        self.imu_msg.header.stamp = self.get_clock().now().to_msg()
        self.imu_msg.header.frame_id = "base_link"
        self.imu_msg.orientation.w = quat[0]
        self.imu_msg.orientation.x = quat[1]
        self.imu_msg.orientation.y = quat[2]
        self.imu_msg.orientation.z = quat[3]
        self.imu_msg.angular_velocity.x = gyro[0]
        self.imu_msg.angular_velocity.y = gyro[1]
        self.imu_msg.angular_velocity.z = gyro[2]
        self.imu_msg.linear_acceleration.x = acce[0]
        self.imu_msg.linear_acceleration.y = acce[1]
        self.imu_msg.linear_acceleration.z = acce[2]
        self.dep_msg.data = dep


    def get_image_raw(self,w,h):
        # 定义视口大小
        viewport = mujoco.MjrRect(0, 0, w, h)
        # 更新场景
        mujoco.mjv_updateScene(
            self.model, self.data, mujoco.MjvOption(), 
            None, self.camera, mujoco.mjtCatBit.mjCAT_ALL, self.scene
        )
        # 渲染到缓冲区
        mujoco.mjr_render(viewport, self.scene, self.mjr_context)
        # 读取 RGB 数据（格式为 HWC, uint8）
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        mujoco.mjr_readPixels(rgb, None, viewport, self.mjr_context)
        cv_image = cv2.cvtColor(np.flipud(rgb), cv2.COLOR_RGB2BGR)
        return cv_image


    def simulator(self):
        # 初始化glfw
        glfw.init()
        glfw.window_hint(glfw.VISIBLE,glfw.FALSE)
        self.window = glfw.create_window(1200,900,"mujoco",None,None)
        glfw.make_context_current(self.window)

        # 创建相机
        self.camera = mujoco.MjvCamera()
        self.camera.fixedcamid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "cam")
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FIXED 
        self.scene = mujoco.MjvScene(self.model, maxgeom=1000)
        self.mjr_context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.mjr_context)

        self.get_logger().info("Simulation thread started")
        while self._running and rclpy.ok():
            try:
                step_start = time.time()
                
                # 控制输入
                with self.control_lock:
                    self.data.ctrl[:] = self.control_data
                
                # 单步更新
                mujoco.mj_step(self.model, self.data)

                # 传感器数据发布
                current_time = time.time()
                if current_time - self.last_pub_sensor_time >= self.pub_sensor_interval:
                    self.get_sensor_data()
                    self.imu_pub.publish(self.imu_msg)
                    self.dep_pub.publish(self.dep_msg)
                    # self.get_logger().info(f"{self.imu_msg}")
                    # self.get_logger().info(f"{self.dep_msg}")
                    self.last_pub_sensor_time = current_time

                # 摄像头图像渲染
                # img = self.get_image_raw(640,480)
                # cv2.imshow("img",img)
                # cv2.waitKey(1)

                # 查看器更新
                if self.viewer and self.viewer.is_running():
                    with self.viewer.lock():
                        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(self.data.time % 2)
                    self.viewer.sync()

                # 实时同步
                time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

            except Exception as e:
                self.get_logger().error(f"Simulation error: {str(e)}")


def main(args=None):
    rclpy.init(args=args)
    node = MuJoCoSimulatorNode("mujoco_simulator_node")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().warn("KeyboardInterrupt, shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()
