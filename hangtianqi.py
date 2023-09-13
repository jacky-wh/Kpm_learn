import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import time

def hat(x):
    xhat=np.array([[0,-x[2],x[1]],[x[2],0,-x[0]],[-x[1],x[0],0]],dtype=object)
    return xhat

#x1=2x2;x2=-0.8x1+2x2-10x1^2x2+u
#定义状态方程（vanderpol）

def Spacecraft(STATE, t, INPUT):
    I=np.diag([1,2,3])
    q=STATE[0:4].reshape(-1,1)
    qn=np.linalg.norm(q)
    q=q/qn
    w=STATE[4:7].reshape(-1,1)
    inv_I = np.linalg.inv(I)
    wdot = np.dot(inv_I, -hat(w) @ (I @ w)+INPUT.reshape(-1,1))
    qdot1 = 0.5 * (hat(q[0:3]) + q[3] * np.eye(3)) @ w
    qdot2 = (-np.transpose(q[0:3]) @ w * 0.5).reshape(-1,1)
    # state=np.concatenate((qdot1,qdot2,wdot),axis=0)
    return [qdot1[0].item(),qdot1[1].item(),qdot1[2].item(),qdot2[0].item(),wdot[0].item(),wdot[1].item(),wdot[2].item()]

pi=np.pi
InitialCondition = [0.,0.,0.,1.,pi/10,pi/8,pi/6]

#尝试代码运行
dt = 0.01
timespan = np.arange(0, 1 + dt, dt)
u_snapshot_train = np.random.rand(3, 1000) * 2 - 1
# Snapshots = odeint(Spacecraft, InitialCondition,timespan,args=(u_snapshot_train[:,0],),).T



dt = 0.05  # 采样间隔
N_state = 7 # 状态量个数
N_sim = 200  # 仿真周期数
N_case = 1000 # 仿真次数

# 随机生成初始条件和输入
x_init_train = np.random.rand(N_state, N_case) * 2 - 1
u_snapshot_train = np.random.rand(3, N_sim * N_case) * 2 - 1

# 可使用0输入检验建模准确性
# u_snapshot_train = np.zeros((1, N_sim * N_case))

x_snapshot_train = np.zeros((N_state, N_case * (N_sim + 1)))

start =time.time()
for i in range(N_case):
    print(i)
    # 导入初始条件
    x_snapshot_train[:, i * N_sim] = x_init_train[:, i]
    for j in range(N_sim):
        # 通过求解微分方程得到状态量数据，论文原代码使用龙格库塔法
        InitialCondition = x_snapshot_train[:, i * N_sim + j].tolist()
        snapshot_temp = odeint(
            Spacecraft,
            InitialCondition,
            np.linspace(0, dt, 101),
            args=(u_snapshot_train[:, i * N_sim + j],),
        ).T
        x_snapshot_train[:, i * N_sim + j + 1] = snapshot_temp[:, -1]

end = time.time()
print('data Running time: %s Seconds'%(end-start))

delete_list_1, delete_list_2 = [], []

for i in range(N_case):
    delete_list_1.append((i + 1) * N_sim )
    delete_list_2.append(i * N_sim)

X = np.delete(x_snapshot_train, delete_list_1, axis=1)
Xprime = np.delete(x_snapshot_train, delete_list_2, axis=1)
Y = u_snapshot_train


def RBF(x, x_center, epsilon=1, k=1, TYPE="ThinPlate"):
    RADIUS = np.linalg.norm(x - x_center)

    TYPE = TYPE.lower()
    if TYPE == "thinplate":
        return 0 if RADIUS == 0 else np.power(RADIUS, 2) * np.log(RADIUS)
    elif TYPE == "gauss":
        return np.exp(-np.power(epsilon * RADIUS, 2))
    elif TYPE == "invquad":
        return 1 / (1 + np.power(epsilon * RADIUS, 2))
    elif TYPE == "invmultquad":
        return 1 / np.sqrt(1 + np.power(epsilon * RADIUS, 2))
    elif TYPE == "polyharmonic":
        return np.power(RADIUS, k) * np.log(RADIUS)


def LiftFun(x, x_center, Type="thinplate", Epsilon=1, K=1, WithOriginalState=True):
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    x_rbf = np.zeros((len(x_center), x.shape[1]))

    for column in range(x_rbf.shape[1]):
        for row in range(len(x_center)):
            x_rbf[row, column] = RBF(
                x[:, column].reshape(-1, 1),
                x_center[row],
                epsilon=Epsilon,
                k=K,
                TYPE=Type,
            )

    if WithOriginalState:
        return np.concatenate((x, x_rbf), axis=0)
    else:
        return x_rbf.reshape(-1, x.shape[1])

N_RBFcenters = 200

centers = np.random.rand(N_state, N_RBFcenters) * 2 - 1
centers_tuple = [centers[:, i] for i in range(centers.shape[1])]
centers_tuple = tuple(centers_tuple)

start =time.time()
Z = LiftFun(X, centers_tuple, Type="thinplate", WithOriginalState=True)
Zprime = LiftFun(Xprime, centers_tuple, Type="thinplate", WithOriginalState=True)
end = time.time()
print('lift Running time: %s Seconds'%(end-start))


def DMDc_Milan(Z, Zprime, Y, X):
    W_left, W_right = (
        np.concatenate((Zprime, X), axis=0) @ np.concatenate((Z, Y), axis=0).conj().T,
        np.concatenate((Z, Y), axis=0) @ np.concatenate((Z, Y), axis=0).conj().T,
    )

    A_B_C_0 = W_left @ np.linalg.pinv(W_right)

    A = A_B_C_0[: Zprime.shape[0], : Z.shape[0]]
    B = A_B_C_0[: Zprime.shape[0], Z.shape[0] :]
    C = A_B_C_0[Zprime.shape[0] :, : Z.shape[0]]
    # 看右下角是不是全0，直接对右下角所有元素的平方求和
    error = np.sum(np.power(A_B_C_0[Zprime.shape[0] :, Z.shape[0] :], 2))

    return A, B, C, error
abc= DMDc_Milan(Z, Zprime, Y, X)
A_Milan, B_Milan, C_Milan = abc[0], abc[1], abc[2]

N_sim_test = 100
x_init_test = np.random.rand(N_state, 1) * 2 - 1
u_snapshot_test = np.random.rand(3, N_sim_test) * 2 - 1
x_snapshot_test_true = np.zeros((N_state, N_sim_test + 1))
x_snapshot_test_true[:, 0] = x_init_test.reshape(-1,)
x_snapshot_test_DMDc_Milan = (
    np.zeros((N_state, N_sim_test + 1))
)

x_snapshot_test_DMDc_Milan[:, 0] = x_init_test.reshape(
    -1,
)


start=time.time()
for j in range(N_sim_test):
    # 通过求解微分方程得到状态量数据，论文原代码使用龙格库塔法
    InitialCondition = x_snapshot_test_true[:, j].tolist()
    snapshot_temp = odeint(
        Spacecraft,
        InitialCondition,
        np.linspace(0, dt, 101),
        args=(u_snapshot_test[:, j],),
    ).T
    x_snapshot_test_true[:, j + 1] = snapshot_temp[:, -1]
end=time.time()
print('real Running time: %s Seconds'%(end-start))

start=time.time()
for j in range(N_sim_test):
    z_temp_DMD_Milan = LiftFun(
        x_snapshot_test_DMDc_Milan[:, j],
        centers_tuple,
        Type="thinplate",
    )

    zprime_temp_DMD_Milan = A_Milan @ z_temp_DMD_Milan + ( B_Milan @ u_snapshot_test[:, j]).reshape(N_state + N_RBFcenters, 1)


    x_Milan_temp = C_Milan @ zprime_temp_DMD_Milan
    x_snapshot_test_DMDc_Milan[:, j + 1] = x_Milan_temp[:, -1]


end=time.time()
print('koopman Running time: %s Seconds'%(end-start))



plt.plot(np.linspace(0, N_sim_test * dt, N_sim_test + 1), x_snapshot_test_true[0, :], label="Ture",)
plt.scatter(
    np.linspace(0, N_sim_test * dt, N_sim_test + 1),
    x_snapshot_test_DMDc_Milan[0, :],
    marker="x",
    c="g",
    label="koopman",
)
plt.legend(loc='upper left', bbox_to_anchor=(0.2, 0.95))
plt.title("True_and_koopman", fontsize=25)
plt.xlabel("$t$", fontsize=13, x=1)
plt.ylabel("$x$", fontsize=13, y=1, rotation=1)
plt.show()

plt.plot(np.linspace(0, N_sim_test * dt, N_sim_test + 1), x_snapshot_test_true[1, :], label="Ture",)
plt.scatter(
    np.linspace(0, N_sim_test * dt, N_sim_test + 1),
    x_snapshot_test_DMDc_Milan[1, :],
    marker="x",
    c="g",
    label="koopman",
)
plt.legend(loc='upper left', bbox_to_anchor=(0.2, 0.95))
plt.title("True_and_koopman", fontsize=25)
plt.xlabel("$t$", fontsize=13, x=1)
plt.ylabel("$x$", fontsize=13, y=1, rotation=1)
plt.show()

plt.plot(np.linspace(0, N_sim_test * dt, N_sim_test + 1), x_snapshot_test_true[2, :], label="Ture",)
plt.scatter(
    np.linspace(0, N_sim_test * dt, N_sim_test + 1),
    x_snapshot_test_DMDc_Milan[2, :],
    marker="x",
    c="g",
    label="koopman",
)
plt.legend(loc='upper left', bbox_to_anchor=(0.2, 0.95))
plt.title("True_and_koopman", fontsize=25)
plt.xlabel("$t$", fontsize=13, x=1)
plt.ylabel("$x$", fontsize=13, y=1, rotation=1)
plt.show()

plt.plot(np.linspace(0, N_sim_test * dt, N_sim_test + 1), x_snapshot_test_true[3, :], label="Ture",)
plt.scatter(
    np.linspace(0, N_sim_test * dt, N_sim_test + 1),
    x_snapshot_test_DMDc_Milan[3, :],
    marker="x",
    c="g",
    label="koopman",
)
plt.legend(loc='upper left', bbox_to_anchor=(0.2, 0.95))
plt.title("True_and_koopman", fontsize=25)
plt.xlabel("$t$", fontsize=13, x=1)
plt.ylabel("$x$", fontsize=13, y=1, rotation=1)
plt.show()

plt.plot(np.linspace(0, N_sim_test * dt, N_sim_test + 1), x_snapshot_test_true[4, :], label="Ture",)
plt.scatter(
    np.linspace(0, N_sim_test * dt, N_sim_test + 1),
    x_snapshot_test_DMDc_Milan[4, :],
    marker="x",
    c="g",
    label="koopman",
)
plt.legend(loc='upper left', bbox_to_anchor=(0.2, 0.95))
plt.title("True_and_koopman", fontsize=25)
plt.xlabel("$t$", fontsize=13, x=1)
plt.ylabel("$x$", fontsize=13, y=1, rotation=1)
plt.show()

plt.plot(np.linspace(0, N_sim_test * dt, N_sim_test + 1), x_snapshot_test_true[5, :], label="Ture",)
plt.scatter(
    np.linspace(0, N_sim_test * dt, N_sim_test + 1),
    x_snapshot_test_DMDc_Milan[5, :],
    marker="x",
    c="g",
    label="koopman",
)
plt.legend(loc='upper left', bbox_to_anchor=(0.2, 0.95))
plt.title("True_and_koopman", fontsize=25)
plt.xlabel("$t$", fontsize=13, x=1)
plt.ylabel("$x$", fontsize=13, y=1, rotation=1)
plt.show()

plt.plot(np.linspace(0, N_sim_test * dt, N_sim_test + 1), x_snapshot_test_true[6, :], label="Ture",)
plt.scatter(
    np.linspace(0, N_sim_test * dt, N_sim_test + 1),
    x_snapshot_test_DMDc_Milan[6, :],
    marker="x",
    c="g",
    label="koopman",
)
plt.legend(loc='upper left', bbox_to_anchor=(0.2, 0.95))
plt.title("True_and_koopman", fontsize=25)
plt.xlabel("$t$", fontsize=13, x=1)
plt.ylabel("$x$", fontsize=13, y=1, rotation=1)
plt.show()


