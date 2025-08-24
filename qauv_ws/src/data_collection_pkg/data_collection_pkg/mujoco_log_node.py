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
from std_msgs.msg import Float64MultiArray
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation


"""
数据采集参数设置(新)
相机高度    相机视野                采样数
0.40       (-0.16,0.04) 0.20     0.0800 800
0.45       (-0.18,0.06) 0.22     0.1056 1000
0.50       (-0.20,0.08) 0.24     0.1344 1300
0.55       (-0.22,0.10) 0.26     0.1664 1650
0.60       (-0.24,0.12) 0.28     0.2016 2000
0.65       (-0.26,0.14) 0.31     0.2480 2500
0.70       (-0.28,0.16) 0.34     0.2992 3000
0.75       (-0.30,0.18) 0.37     0.3552 3500
0.80       (-0.32,0.20) 0.40     0.4160 4000
area_d: (1000(0.8),9000(0.4))
"""


class MuJoCoLogNode(Node):
    def __init__(self, name):
        super().__init__(name)
        
        # 主线程运行标志
        self._running = True

        # 查看器就绪标志
        self.viewer_ready = threading.Event()

        # 创建订阅和发布
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.expert_pub = self.create_publisher(Float64MultiArray, '/expert_data', qos)

        # 发布消息的格式与尺寸
        self.expert_msg = Float64MultiArray()
        self.expert_msg.data = [0.0] * 6
        
        # OpenCV/ROS图像转换
        self.bridge = CvBridge()

        # OpenCV 红色HSV阈值
        self.lower_red = np.array([0, 100, 100])
        self.upper_red = np.array([10, 255, 255])

        # 设置发布频率
        self.pub_rate = 5.0
        self.pub_interval = 1.0 / self.pub_rate
        self.last_pub_time = time.time()

        # 创建仿真环境
        self.model = mujoco.MjModel.from_xml_path('/home/zb/qauv_ws/src/data_collection_pkg/data_collection_pkg/data_collection.xml')
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

        # 图像渲染相关
        self.window = None
        self.camera = None
        self.scene = None
        self.mjr_context = None

        # MUJOCO ID与地址
        # 跟随相机
        self.cam_tracking_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "cam_tarcking")
        # 目标
        self.object_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,"object")
        self.object_qpos_addr = self.model.jnt_qposadr[self.model.body_jntadr[self.object_id]]
        # 机器人
        self.qauv_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,"qauv")
        self.qauv_qpos_addr = self.model.jnt_qposadr[self.model.body_jntadr[self.qauv_id]]

        # 启动仿真线程
        self.sim_thread.start()
        
        self.get_logger().info("MuJoCo Log Node Started")


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

    
    def get_euler_data(self):
        cam_rot_matrix = self.data.cam_xmat[self.cam_tracking_id].reshape(3, 3)
        euler_rad = Rotation.from_matrix(cam_rot_matrix).as_euler('zyx') # 返回数据依次是绕z, y, x轴的旋转角度
        self.expert_msg.data[0] = euler_rad[2] # roll
        self.expert_msg.data[1] = euler_rad[1] # pitch


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
    

    def object_detect(self, cv_image):
        """
        最小外接矩阵
        """
        # BGR转HSV
        hsv_img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        # 图像二值化
        mask_red = cv2.inRange(hsv_img, self.lower_red, self.upper_red)
        # 目标轮廓检测
        contours, hierarchy = cv2.findContours(mask_red, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        if len(contours) > 0:
            # 找到面积最大的轮廓
            max_contour = max(contours, key=cv2.contourArea)
            # 绘制最小外接矩形, 中心点
            rotated_rect = cv2.minAreaRect(max_contour)
            (center_x, center_y), (width, height), angle = rotated_rect
            box_points = cv2.boxPoints(rotated_rect)
            box_points = np.int0(box_points)
            cv2.drawContours(cv_image, [box_points], 0, (0, 255, 0), 1)
            cv2.circle(cv_image, (int(center_x), int(center_y)), 1, (0, 255, 0), -1)
            self.expert_msg.data[2] = 320 - center_x
            self.expert_msg.data[3] = 240 - center_y
            self.expert_msg.data[4] = width*height
            self.expert_msg.data[5] = angle
        else:
            self.expert_msg.data[2] = 0.0
            self.expert_msg.data[3] = 0.0
            self.expert_msg.data[4] = 0.0
            self.expert_msg.data[5] = 0.0
        return cv_image


    def simulator(self):
        # 初始化glfw
        glfw.init()
        glfw.window_hint(glfw.VISIBLE,glfw.FALSE)
        self.window = glfw.create_window(1200,900,"mujoco",None,None)
        glfw.make_context_current(self.window)

        # 创建相机
        self.camera = mujoco.MjvCamera()
        self.camera.fixedcamid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "cam_fixed")
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.scene = mujoco.MjvScene(self.model, maxgeom=1000)
        self.mjr_context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.mjr_context)

        self.get_logger().info("Simulation thread started")
        while self._running and rclpy.ok():
            try:
                step_start = time.time()
                
                # 单步更新
                mujoco.mj_step(self.model, self.data)

                # 摄像头图像获取, 目标检测, 渲染
                # img = self.get_image_raw(640,480)
                # img = self.object_detect(img)
                # cv2.imshow("img",img)
                # cv2.waitKey(1)

                # 数据发布
                current_time = time.time()
                if current_time - self.last_pub_time >= self.pub_interval:
                    # 像素-角度计算
                    img = self.get_image_raw(640,480)
                    img = self.object_detect(img)
                    self.get_euler_data()
                    # 消息发布
                    self.expert_pub.publish(self.expert_msg)
                    # 位置测量
                    # pos = self.data.xpos[self.object_id]
                    # self.get_logger().info(f"pos: {pos}")
                    # 位置更新
                    object_pos = [np.random.uniform(-0.32,0.20), np.random.uniform(-0.40,0.40), 0.02]
                    self.data.qpos[self.object_qpos_addr:self.object_qpos_addr+3] = object_pos
                    qauv_pos = [0, 0, np.random.uniform(0.4,0.8)]
                    self.data.qpos[self.qauv_qpos_addr:self.qauv_qpos_addr+3] = qauv_pos
                    self.last_pub_time = current_time

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
    node = MuJoCoLogNode("mujoco_log_node")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().warn("KeyboardInterrupt, shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()
