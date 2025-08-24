#!/home/zb/anaconda3/envs/mujoco/bin/python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
import time
import collections
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Float64MultiArray
import csv


class DataLogNode(Node):
    def __init__(self, name):
        super().__init__(name)
        
        # 数据缓存（使用 deque 作为循环缓冲区）
        self.para_buffer = collections.deque(maxlen=50)
        self.control_buffer = collections.deque(maxlen=50)
        self.error_buffer = collections.deque(maxlen=50)
        
        # 创建订阅
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        self.para_sub = self.create_subscription(Float32MultiArray, '/para_mujoco_data', self.para_callback, qos)
        self.control_sub = self.create_subscription(Float64MultiArray, '/control_cmd', self.control_callback, qos)
        self.error_sub = self.create_subscription(Float32MultiArray, '/error', self.error_callback, qos)
        
        # 获取时间戳
        timestamp_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        self.file_name = f"data_log_{timestamp_str}.csv"
        
        # 打开CSV文件并写入表头
        self.file = open(self.file_name, mode='w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow(['timestamp',
                            'obj_x', 'obj_y', 'obj_z', 'qauv_x', 'qauv_y', 'qauv_z', 'qauv_qw', 'qauv_qx', 'qauv_qy', 'qauv_qz', 
                            'force_1', 'force_2', 'force_3', 'force_4', 
                            'u_e', 'v_e', 'a_e', 'a_c', 'roll_delta_d', 'pitch_delta_d'])

        # 创建定时器
        self.timer = self.create_timer(0.02, self.log_data)
        
        self.get_logger().info("Data Log Node Started!")


    def para_callback(self, msg: Float32MultiArray):
        timestamp = time.time()
        self.para_buffer.append((timestamp, msg.data))


    def control_callback(self, msg: Float64MultiArray):
        timestamp = time.time()
        self.control_buffer.append((timestamp, msg.data))


    def error_callback(self, msg: Float32MultiArray):
        timestamp = time.time()
        self.error_buffer.append((timestamp, msg.data))


    def log_data(self):
        try:
            if not (self.para_buffer and self.control_buffer and self.error_buffer):
                return
            
            # 获取最新的数据时间戳
            current_time = time.time()
            
            # 查找时间上最接近的数据（在0.1秒窗口内）
            para_data = self.find_closest_data(self.para_buffer, current_time)
            control_data = self.find_closest_data(self.control_buffer, current_time)
            error_data = self.find_closest_data(self.error_buffer, current_time)
            
            if para_data and control_data and error_data:
                para_timestamp, para_values = para_data
                control_timestamp, control_values = control_data
                error_timestamp, error_values = error_data
                
                # 检查时间一致性（所有数据时间差在0.05秒内）
                max_time_diff = max(abs(para_timestamp - control_timestamp),
                                   abs(para_timestamp - error_timestamp),
                                   abs(control_timestamp - error_timestamp))
                
                if max_time_diff < 0.03:  # 30ms 容差
                    # 提取数据
                    obj_x = para_values[0]
                    obj_y = para_values[1]
                    obj_z = para_values[2]
                    qauv_x = para_values[3]
                    qauv_y = para_values[4]
                    qauv_z = para_values[5]
                    qauv_qw = para_values[6]
                    qauv_qx = para_values[7]
                    qauv_qy = para_values[8]
                    qauv_qz = para_values[9]

                    force_1 = control_values[0]
                    force_2 = control_values[1]
                    force_3 = control_values[2]
                    force_4 = control_values[3]

                    u_e = error_values[0]
                    v_e = error_values[1]
                    a_e = error_values[2]
                    a_c = error_values[3]
                    roll_delta_d = error_values[4]
                    pitch_delta_d = error_values[5]
                        
                    # 使用最早的时间戳作为记录时间
                    record_time = min(para_timestamp, control_timestamp, error_timestamp)
                    
                    # 写入 CSV
                    self.writer.writerow([record_time,
                                         obj_x, obj_y, obj_z, qauv_x, qauv_y, qauv_z, qauv_qw, qauv_qx, qauv_qy, qauv_qz, 
                                         force_1, force_2, force_3, force_4, 
                                         u_e, v_e, a_e, a_c, roll_delta_d, pitch_delta_d])
                    
                    self.file.flush()
                
        except Exception as e:
            self.get_logger().error(f"Data alignment error: {e}")


    def find_closest_data(self, buffer, target_time):
        """在缓冲区中查找时间上最接近的数据"""
        if not buffer:
            return None
        
        # 简单返回最新的数据（对于高频数据通常足够）
        return buffer[-1]
        
        # 或者使用更精确的查找（如果需要更严格的时间对齐）
        # closest_data = None
        # min_diff = float('inf')
        # for timestamp, data in buffer:
        #     time_diff = abs(timestamp - target_time)
        #     if time_diff < min_diff:
        #         min_diff = time_diff
        #         closest_data = (timestamp, data)
        # return closest_data if min_diff < 0.1 else None  # 100ms 容差


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
        