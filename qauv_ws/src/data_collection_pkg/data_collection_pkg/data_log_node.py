#!/home/zb/anaconda3/envs/mujoco/bin/python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
import time
import threading
import numpy as np
from std_msgs.msg import Float64MultiArray
import csv


class DataLogNode(Node):
    def __init__(self, name):
        super().__init__(name)
        
        # 数据, 数据锁
        self.latest_expert_data = None
        self.expert_data_lock = threading.Lock()
        
        # 创建订阅和发布
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.expert_sub = self.create_subscription(Float64MultiArray, '/expert_data', self.expert_callback, qos)
        
        # 获取时间戳
        timestamp_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        self.file_name = f"data_log_{timestamp_str}.csv"
        
        # 打开CSV文件并写入表头
        self.file = open(self.file_name, mode='w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow(['timestamp',
                            'roll_delta_d', 'pitch_delta_d', 
                            'u_e', 'v_e', 'a_c'])
        
        # 创建定时器
        self.timer = self.create_timer(0.2, self.log_data)

        self.get_logger().info("Data Log Node Started!")


    def expert_callback(self, msg: Float64MultiArray):
        with self.expert_data_lock:
            self.latest_expert_data = msg.data
    
    
    def log_data(self):
        try:
            with self.expert_data_lock:
                expert_data = self.latest_expert_data

            # 获取时间戳
            timestamp = self.get_clock().now().nanoseconds / 1e9
                
            # 提取图像数据
            if expert_data is not None:
                roll_delta_d = expert_data[0]
                pitch_delta_d = expert_data[1]
                u_e = expert_data[2]
                v_e = expert_data[3]
                a_c = expert_data[4]
            else:
                roll_delta_d, pitch_delta_d, u_e, v_e, a_c = None, None, None, None, None
                
            # Write to CSV
            self.writer.writerow([timestamp,
                                 roll_delta_d, pitch_delta_d, 
                                 u_e, v_e, a_c])
        
        except Exception as e:
            self.get_logger().error(f"Data log error: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = DataLogNode("data_log_node")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().warn("KeyboardInterrupt, shutting down...")
    finally:
        node.file.close()
        node.destroy_node()
        rclpy.shutdown()
