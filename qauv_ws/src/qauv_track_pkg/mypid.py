class MyPID:
    def __init__(self, kp=1.0, ki=0.0, kd=0.0, limit_iout=None, limit_out_dlt=None, limit_out=None):
        """
        PID初始化
        :param kp: 比例系数
        :param ki: 积分系数
        :param kd: 微分系数
        :param limit_iout: 积分输出限幅
        :param limit_out_dlt: 输出增量限幅
        :param limit_out: 总输出限幅
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.limit_iout = limit_iout
        self.limit_out_dlt = limit_out_dlt
        self.limit_out = limit_out

        self.pout = 0.0  # P输出
        self.iout = 0.0  # I输出
        self.dout = 0.0  # D输出
        self.out_dlt = 0.0  # 输出增量
        self.out = 0.0  # 总输出

        self.error_now = 0.0  # 当前误差
        self.error_last = 0.0  # 上一次误差
        self.error_prev = 0.0  # 上上次误差

    def pid_delta(self, error):
        """
        增量式PID
        :param ref: 目标值
        :param real: 实际值
        :return: 总输出
        """
        # 误差计算
        self.error_now = error
        # PID输出计算
        self.pout = self.kp * (self.error_now - self.error_last)
        self.iout = self.ki * self.error_now
        self.dout = self.kd * (self.error_now - 2 * self.error_last + self.error_prev)
        # 输出增量计算
        self.out_dlt = self.pout + self.iout + self.dout
        # 输出增量限幅
        if self.limit_out_dlt is not None:
            self.out_dlt = min(max(self.out_dlt, -self.limit_out_dlt), self.limit_out_dlt)
        # 总输出计算
        self.out += self.out_dlt
        # 总输出限幅
        if self.limit_out is not None:
            self.out = min(max(self.out, -self.limit_out), self.limit_out)
        # 误差更新
        self.error_prev = self.error_last
        self.error_last = self.error_now
        return self.out

    def pid_position(self, error):
        """
        位置式PID
        :param ref: 目标值
        :param real: 实际值
        :return: 总输出
        """
        # 误差计算
        self.error_now = error
        # PID输出计算
        self.pout = self.kp * self.error_now
        self.iout += self.ki * self.error_now
        self.dout = self.kd * (self.error_now - self.error_last)
        # 积分限幅
        if self.limit_iout is not None:
            self.iout = min(max(self.iout, -self.limit_iout), self.limit_iout)
        # 总输出计算
        self.out = self.pout + self.iout + self.dout
        # 总输出限幅
        if self.limit_out is not None:
            self.out = min(max(self.out, -self.limit_out), self.limit_out)
        # 误差更新
        self.error_last = self.error_now
        return self.out
