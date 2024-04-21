import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)

# 水平的r z_coordinate phi v_r w_theta w_phi
dim_x = 6
dim_z = 6

# DOA parameters
dt = 1e-2
c = 340
radius = 1
beta = 2 * np.pi / dim_z

# create kalman filter parameters
z = np.zeros(shape=(dim_z, 1), dtype=float)  # 观测值
z_pred = np.zeros(shape=(dim_z, 1), dtype=float)  # 预测的观测值
x_ = np.zeros(shape=(dim_x, 1), dtype=float)  # 先验估计值
P_ = np.zeros(shape=(dim_x, dim_x), dtype=float)  # 先验估计协方差
K = np.zeros(shape=(dim_x, dim_z), dtype=float)  # kalman增益
I = np.eye(N=dim_x, dtype=float)  # 单位矩阵
H = np.zeros(shape=(dim_z, dim_x), dtype=float)  # 观测值的状态转移矩阵
F = np.array([[1, 0, 0, dt, 0, 0],
              [0, 1, 0, 0, dt, 0],
              [0, 0, 1, 0, 0, dt],
              [0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 1]], dtype=float)  # 先验估计值的状态转移矩阵

R_param = 1e-8
R = np.array([[R_param, 0, 0, 0, 0, 0],
              [0, R_param, 0, 0, 0, 0],
              [0, 0, R_param, 0, 0, 0],
              [0, 0, 0, R_param, 0, 0],
              [0, 0, 0, 0, R_param, 0],
              [0, 0, 0, 0, 0, R_param]], dtype=float)  # 观测噪声的方差

Q_param = 0.001
Q = np.array([[Q_param, 0, 0, 0, 0, 0],
              [0, Q_param, 0, 0, 0, 0],
              [0, 0, Q_param, 0, 0, 0],
              [0, 0, 0, Q_param, 0, 0],
              [0, 0, 0, 0, Q_param, 0],
              [0, 0, 0, 0, 0, Q_param]], dtype=float)  # 系统噪声的方差
P = np.ones(shape=(dim_x, dim_x), dtype=float)  # 后验估计协方差
x = np.array([[1],
              [0.1],
              [0],
              [0],
              [0],
              [0]], dtype=float)  # 后验估计值


# 观测方程
@ti.kernel
def h(i: ti.i32):
    loss[None] = (ti.sqrt(rr[None] ** 2 + radius ** 2 + zz[None] ** 2 - 2 * radius *
                          rr[None] * ti.cos(i * beta - ff[None])) - ti.sqrt(rr[None] ** 2 + zz[None] ** 2)) / c


# for gradient
loss = ti.field(ti.f32, shape=(), needs_grad=True)
rr = ti.field(ti.f32, shape=(), needs_grad=True)
zz = ti.field(ti.f32, shape=(), needs_grad=True)
ff = ti.field(ti.f32, shape=(), needs_grad=True)

# 储存模拟值
rr_list = []
zz_list = []
ff_list = []


# generating obs
def gen_Obs(t):
    ret = np.zeros(shape=(dim_z, 1), dtype=float)
    for i in range(dim_z):
        loss[None] = 0
        rr[None] = 2 + t * np.sin(2 * np.pi * t)
        zz[None] = 4 + t * np.sin(2 * np.pi * t)
        ff[None] = t * 2 * np.pi
        with ti.Tape(loss=loss):
            h(i)
        ret[i, 0] = loss[None]

    temp_rr = 2 + t * np.sin(2 * np.pi * t)
    temp_zz = 4 + t * np.sin(2 * np.pi * t)
    temp_ff = t * 2 * np.pi

    rr_list.append(temp_rr)
    zz_list.append(temp_zz)
    ff_list.append(temp_ff)

    return ret + np.random.normal(0, 1e-4, size=(dim_z, 1))


# obs and results
obs = []
results = []
sample_num = 200
for j in range(sample_num):
    test_data = gen_Obs(dt * j)
    obs.append(test_data)


# kalman predict
def EKL_predict():
    global x_, P_
    x_ = F @ x
    P_ = (F @ P) @ F.T + Q


# kalman update H matrix
def update_H():
    global H
    for i in range(dim_z):
        loss[None] = 0
        rr[None] = x_[0, 0]
        zz[None] = x_[1, 0]
        ff[None] = x_[2, 0]
        with ti.Tape(loss=loss):
            h(i)
        z_pred[i, 0] = loss[None]
        H[i] = [rr.grad[None], zz.grad[None], ff.grad[None], 0, 0, 0]


# kalman update
def EKL_update():
    global K, x, P
    K = (P_ @ H.T) @ np.linalg.pinv((H @ P_) @ H.T + R)
    x = x_ + K @ (z - z_pred)
    P = (I - K @ H) @ P_


def main():
    print("Sound localization system")
    global z
    for i in range(sample_num):
        z = np.array(obs[i], dtype=float)
        EKL_predict()
        update_H()
        EKL_update()

        results.append([x[0, 0], x[1, 0], x[2, 0]])

    x1 = results[sample_num - 1]
    print(x1)
    gui = ti.GUI("Sound localization system GUI", background_color=0xDDDDDD)
    while gui.running:
        for i in range(sample_num):
            # 模拟值
            gui.line((i / sample_num, rr_list[i] / (2 * 10) + 0.1),
                     ((i - 1) / sample_num, rr_list[(i - 1)] / (2 * 10) + 0.1),
                     color=0xFF0000, radius=2)  # 红
            gui.line((i / sample_num, zz_list[i] / (2 * 10) + 0.4),
                     ((i - 1) / sample_num, zz_list[(i - 1)] / (2 * 10) + 0.4),
                     color=0x00FF00, radius=2)  # 绿
            gui.line((i / sample_num, ff_list[i] / (2 * 10) + 0.4),
                     ((i - 1) / sample_num, ff_list[(i - 1)] / (2 * 10) + 0.4),
                     color=0x0000FF, radius=2)  # 蓝
            # 预测值
            gui.line((i / sample_num, results[i][0] / (2 * 10) + 0.1),
                     ((i - 1) / sample_num, results[(i - 1)][0] / (2 * 10) + 0.1),
                     color=0xFF00FF, radius=2)  # 紫
            gui.line((i / sample_num, results[i][1] / (2 * 10) + 0.4),
                     ((i - 1) / sample_num, results[(i - 1)][1] / (2 * 10) + 0.4),
                     color=0xFFFF00, radius=2)  # 黄
            gui.line((i / sample_num, results[i][2] / (2 * 10) + 0.4),
                     ((i - 1) / sample_num, results[(i - 1)][2] / (2 * 10) + 0.4),
                     color=0x00FFFF, radius=2)  # 天蓝
        gui.show()


if __name__ == '__main__':
    main()
