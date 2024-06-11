import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Constants
a0 = 1  # Bohr radius, for simplicity, set to 1

# 각 부분 정의
# GPT 왈 : 이 함수는 궤도 각운동량 양자수 𝑙, 자기 양자수 𝑚 극좌표계의 각도 𝜃 및 𝜙를 입력받아 각 부분을 계산합니다. 이는 구면 조화 함수 𝑌𝑙𝑚를 계산하는 과정입니다.
def angular_part(l, m, theta, phi):
    if l == 0:  # s-orbital
        return (1 / (4 * np.pi))**0.5
    elif l == 1:  # p-orbital
        if m == 0:
            return (3 / (4 * np.pi))**0.5 * np.cos(theta)
        elif m == 1:
            return (3 / (4 * np.pi))**0.5 * np.sin(theta) * np.cos(phi)
        elif m == -1:
            return (3 / (4 * np.pi))**0.5 * np.sin(theta) * np.sin(phi)
    elif l == 2:  # d-orbital
        if m == 0:
            return (5 / (4 * np.pi))**0.5 * (3 * np.cos(theta)**2 - 1)
        elif m == 1:
            return (15 / (4 * np.pi))**0.5 * np.cos(theta) * np.sin(theta) * np.cos(phi)
        elif m == -1:
            return (15 / (4 * np.pi))**0.5 * np.cos(theta) * np.sin(theta) * np.sin(phi)
        elif m == 2:
            return (15 / (4 * np.pi))**0.5 * np.sin(theta)**2 * np.cos(2 * phi)
        elif m == -2:
            return (15 / (4 * np.pi))**0.5 * np.sin(theta)**2 * np.sin(2 * phi)
    elif l == 3: # f-orbital
        if m == 0:
            return (7 / (16 * np.pi)) * (5 * np.cos(theta)**3 - 3 * np.cos(theta))
        elif m == 1:
            return (21 / (32 * np.pi)) * np.sin(theta) * (5 * np.cos(theta)**2 - 1) * np.cos(phi)
        elif m == -1:
            return (21 / (32 * np.pi)) * np.sin(theta) * (5 * np.cos(theta)**2 - 1) * np.sin(phi)
        elif m == 2:
            return (105 / (4 * np.pi)) * np.sin(theta)**2 * np.cos(theta) * np.cos(2 * phi)
        elif m == -2:
            return (105 / (4 * np.pi)) * np.sin(theta)**2 * np.cos(theta) * np.sin(2 * phi)
        elif m == 3:
            return (35 / (32 * np.pi)) * np.sin(theta)**3 * np.cos(3 * phi)
        elif m == -3:
            return (35 / (32 * np.pi)) * np.sin(theta)**3 * np.sin(3 * phi)
    else:
        raise ValueError(f"Unsupported l={l} or m={m}")

# 방사 부분 정의
def radial_part(n, l, r):
    if n == 1 and l == 0:
        return 2 * (1 / a0)**(3/2) * np.exp(-r / a0)
    elif n == 2 and l == 0:
        return (1 / (2 * a0)**(3/2)) * (2 - r / a0) * np.exp(-r / (2 * a0))
    elif n == 2 and l == 1:
        return (1 / (24 * a0**3))**0.5 * (r / a0) * np.exp(-r / (2 * a0))
    elif n == 3 and l == 0:
        return (2 / (27 * a0**3))**0.5 * (27 - 18 * (r / a0) + 2 * (r / a0)**2) * np.exp(-r / (3 * a0))
    elif n == 3 and l == 1:
        return (8 / (27 * a0**3))**0.5 * (1 - (r / (6 * a0))) * (r / a0) * np.exp(-r / (3 * a0))
    elif n == 3 and l == 2:
        return (1 / (81 * a0**3))**0.5 * (r / a0)**2 * np.exp(-r / (3 * a0))
    elif n == 4 and l == 3:
        return (4 / (3 * (2 * a0)**3))**0.5 * (1 - 3 * (r / (4 * a0)) + (3/2) * (r / (4 * a0))**2 - (r / (8 * a0))**3) * np.exp(-r / (4 * a0))
    else:
        raise ValueError(f"Unsupported n={n} or l={l}")


# 확률 밀도 함수 계산
def probability_density(n, l, m, r, theta, phi):
    R = radial_part(n, l, r)
    Y = angular_part(l, m, theta, phi)
    return (R * Y)**2

# 무작위 점 생성 후 구 좌표계에서 분포시킴
N = 10000
r = np.random.exponential(scale=5*a0, size=N)
theta = np.random.uniform(0, np.pi, N)
phi = np.random.uniform(0, 2*np.pi, N)

# 구 좌표를 직교 좌표로 변환
x = r * np.sin(theta) * np.cos(phi)
y = r * np.sin(theta) * np.sin(phi)
z = r * np.cos(theta)

def plot_orbital(n, l, m):
    # Calculate the probability density for each point
    prob_density = probability_density(n, l, m, r, theta, phi)
    
    # Normalize the probability density to use as point size
    sizes = (prob_density / np.max(prob_density)) * 50
    
    # Plot the points
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(x, y, z, c=prob_density, cmap='plasma', s=sizes)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'{n}{["s", "p", "d", "f"][l]}_{m} Orbital')
    
    plt.colorbar(sc)
    plt.show()
    # fig.savefig(f'image/plot_orbital({n}_{l}_{m}).png')

# Plot s, p, d orbitals
plot_orbital(1, 0, 0)  # 1s orbital
for m in range(0, 2):
    plot_orbital(2, 1, m)  # 2p_z orbital
for m in range(0, 3):
    plot_orbital(3, 2, m)  # 3d_z^2 orbital
for m in range(0, 4):
    plot_orbital(4, 3, m)  # 4f_z^3 orbital