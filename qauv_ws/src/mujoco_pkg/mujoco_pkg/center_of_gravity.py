import mujoco
import numpy as np

# 加载模型和创建数据
model = mujoco.MjModel.from_xml_path('/home/zb/qauv_ws/src/mujoco_pkg/mujoco_pkg/qauv_track.xml')
data = mujoco.MjData(model)

# 运行前向动力学（确保状态更新）
mujoco.mj_forward(model, data)

# 获取整个模型的 COM（索引 0 对应的是世界坐标系下的整体 COM）
com = data.subtree_com[0]
print("Center of Mass (COM):", com)
