import math
import matplotlib.pyplot as plt


class RectanglePath:
    def __init__(self, x_min, y_min, x_max, y_max, v, dt):
        self.x_min, self.y_min = x_min, y_min
        self.x_max, self.y_max = x_max, y_max
        self.v = v
        self.dt = dt
        self.x, self.y = x_min, y_min
        # self.x, self.y = -x_min, y_min
        self.direction = 0

    def update_position(self):
        if self.direction == 0: # x 增大
            self.x += self.v * self.dt
            if self.x >= self.x_max:
                self.x = self.x_max
                self.direction = 1
        elif self.direction == 1: # y 增大
            self.y += self.v * self.dt
            if self.y >= self.y_max:
                self.y = self.y_max
                self.direction = 2
        elif self.direction == 2: # x 减小
            self.x -= self.v * self.dt
            if self.x <= self.x_min:
                self.x = self.x_min
                self.direction = 3
        elif self.direction == 3: # y 减小
            self.y -= self.v * self.dt
            if self.y <= self.y_min:
                self.y = self.y_min
                self.direction = 4
        elif self.direction == 4: # 停止
            self.x, self.y = self.x_min, self.y_min
        return self.x, self.y
    
    # def update_position(self):
    #     if self.direction == 0: # x 减小
    #         self.x -= self.v * self.dt
    #         if self.x <= -self.x_max:
    #             self.x = -self.x_max
    #             self.direction = 1
    #     elif self.direction == 1: # y 增大
    #         self.y += self.v * self.dt
    #         if self.y >= self.y_max:
    #             self.y = self.y_max
    #             self.direction = 2
    #     elif self.direction == 2: # x 增大
    #         self.x += self.v * self.dt
    #         if self.x >= -self.x_min:
    #             self.x = -self.x_min
    #             self.direction = 3
    #     elif self.direction == 3: # y 减小
    #         self.y -= self.v * self.dt
    #         if self.y <= self.y_min:
    #             self.y = self.y_min
    #             self.direction = 0
    #     return self.x, self.y


class CircularPath:
    def __init__(self, x_0, y_0, r, v, dt):
        self.x_0, self.y_0 = x_0, y_0
        self.r = r
        self.v = v
        self.dt = dt
        self.x, self.y = x_0, y_0
        self.angle = 0
        self.direction = 0

    def update_position(self):
        if self.direction == 0: # 逆时针
            self.angle += self.v * self.dt / self.r
            if self.angle >= math.pi * 2:
                self.angle = 0
                self.direction = 4
            self.x = self.x_0 + self.r * math.sin(self.angle)
            self.y = self.y_0 + self.r - self.r * math.cos(self.angle)
        elif self.direction == 4: # 停止
            self.x, self.y = self.x_0, self.y_0
        return self.x, self.y
    
    # def update_position(self):
    #     if self.direction == 0: # 顺时针
    #         self.angle += self.v * self.dt / self.r
    #         if self.angle >= math.pi * 2:
    #             self.angle = 0
    #             self.direction = 4
    #         self.x = self.x_0 - self.r * math.sin(self.angle)
    #         self.y = self.y_0 + self.r - self.r * math.cos(self.angle)
    #     elif self.direction == 4: # 停止
    #         self.x, self.y = self.x_0, self.y_0
    #     return self.x, self.y


class FigureEightPath:
    def __init__(self, x_0, y_0, r, v, dt):
        self.x_0, self.y_0 = x_0, y_0
        self.r = r
        self.v = v
        self.dt = dt
        self.x, self.y = x_0, y_0
        self.angle = 0
        self.direction = 0

    def update_position(self):
        if self.direction == 0: # y轴右半部分, 逆时针
            self.angle += self.v * self.dt / self.r
            if self.angle >= math.pi * 2:
                self.angle = 0
                self.direction = 1
            self.x = self.x_0 + self.r - self.r * math.cos(self.angle)
            self.y = self.y_0 - self.r * math.sin(self.angle)
        elif self.direction == 1: # y轴左半部分, 顺时针
            self.angle += self.v * self.dt / self.r
            if self.angle >= math.pi * 2:
                self.angle = 0
                self.direction = 4
            self.x = self.x_0 - self.r + self.r * math.cos(self.angle)
            self.y = self.y_0 - self.r * math.sin(self.angle)
        elif self.direction == 4: # 停止
            self.x, self.y = self.x_0, self.y_0
        return self.x, self.y


class HelixPath:
    def __init__(self, x_0, y_0, z_0, r, pitch, v, dt):
        self.x_0, self.y_0, self.z_0 = x_0, y_0, z_0
        self.r = r
        self.pitch = pitch
        self.v = v
        self.dt = dt
        self.x, self.y, self.z = x_0, y_0, z_0
        self.angle = 0

    def update_position(self):
        if self.z <= 2 * self.pitch:
            self.angle += self.v * self.dt / self.r
            self.x = self.r * math.cos(self.angle) + self.x_0 - self.r
            self.y = self.r * math.sin(self.angle) + self.y_0
            self.z += (self.v * self.dt * self.pitch) / (2 * math.pi * self.r)
        return self.x, self.y, self.z


# 示例调用
if __name__ == "__main__":
    rect_path = RectanglePath(x_min=0, y_min=0, x_max=5, y_max=5, v=1, dt=0.01)
    fig8_path = FigureEightPath(x_0=0, y_0=0, r=1, v=1, dt=0.01)
    helix_path = HelixPath(x_0=0.5, y_0=0, z_0=0, r=0.5, pitch=1, v=1, dt=0.01)

    # 初始化绘图
    # 初始化绘图
    fig = plt.figure(figsize=(15, 5))
    axs = [fig.add_subplot(131), fig.add_subplot(132), fig.add_subplot(133, projection='3d')]

    axs[0].set_title("rect_path")
    axs[1].set_title("fig8_path")
    axs[2].set_title("helix_path")
    axs[2].set_xlabel('X')
    axs[2].set_ylabel('Y')
    axs[2].set_zlabel('Z')

    # 初始化数据
    rect_x, rect_y = [], []
    rect_x.append(rect_path.x_min)
    rect_y.append(rect_path.y_min)
    fig8_x, fig8_y = [], []
    fig8_x.append(fig8_path.x_0)
    fig8_y.append(fig8_path.y_0)
    helix_x, helix_y, helix_z = [], [], []
    helix_x.append(helix_path.x_0)
    helix_y.append(helix_path.y_0)
    helix_z.append(helix_path.z_0)

    # 仿真时间
    sim_time = 0
    while sim_time < 20:
        # 更新并记录矩形路径位置
        x, y = rect_path.update_position()
        rect_x.append(x)
        rect_y.append(y)

        # 更新并记录“8”字路径位置
        x, y = fig8_path.update_position()
        fig8_x.append(x)
        fig8_y.append(y)

        # 更新并记录螺旋线路径位置
        x, y, z = helix_path.update_position()
        helix_x.append(x)
        helix_y.append(y)
        helix_z.append(z)

        # 增加仿真时间
        sim_time += rect_path.dt

    # 绘制矩形路径
    axs[0].plot(rect_x, rect_y, label="rect_path")
    axs[0].set_xlim([-0.5, 5.5])
    axs[0].set_ylim([-0.5, 5.5])
    axs[0].set_xlabel('X')
    axs[0].set_ylabel('Y')
    axs[0].legend()

    # 绘制“8”字路径
    axs[1].plot(fig8_x, fig8_y, label="fig8_path")
    axs[1].set_xlim([-2.5, 2.5])
    axs[1].set_ylim([-2.5, 2.5])
    axs[1].set_xlabel('X')
    axs[1].set_ylabel('Y')
    axs[1].legend()

    # 绘制螺旋线路径
    axs[2].plot3D(helix_x, helix_y, helix_z, label="helix_path")
    axs[2].set_xlim([-1.5, 1.5])
    axs[2].set_ylim([-1.5, 1.5])
    axs[2].set_xlabel('X')
    axs[2].set_ylabel('Y')
    axs[2].legend()

    plt.tight_layout()
    plt.show()
