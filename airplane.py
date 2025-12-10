import numpy as np  # работа с линейной алгеброй (вектора, матрицы...)
import matplotlib.pyplot as plt  # графика и 3D-визуализация

airplane = np.array([
    [1.5, 0, 0],      # nose
    [-1, -1, 0],      # left wing tip
    [-1, 1, 0],       # right wing tip
    [-0.5, -0.5, 0],  # left inner wing
    [-0.5, 0.5, 0],   # right inner wing
    [-1.2, 0, 0],  # tail base
    [-0.9, 0, 0],    # tail tip
    [-0.8, -0.2, 0],# cockpit left
    [-0.8, 0.2, 0], # cockpit right
    [-1.3, 0, 0.7]      # vertical tail fin
]).T  # shape (3,10)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def draw(points):
    ax.clear()
    ax.scatter(points[0], points[1], points[2], c='black', s=50)
    # линии между ключевыми точками
    ax.plot([points[0,0], points[0,1]], [points[1,0], points[1,1]], [points[2,0], points[2,1]], c='gray')  # nose → left wing
    ax.plot([points[0,0], points[0,2]], [points[1,0], points[1,2]], [points[2,0], points[2,2]], c='gray')  # nose → right wing
    ax.plot([points[0,5], points[0,6]], [points[1,5], points[1,6]], [points[2,5], points[2,6]], c='gray')  # tail
    ax.plot([points[0,6], points[0,9]], [points[1,6], points[1,9]], [points[2,6], points[2,9]], c='gray')  # vertical fin
    ax.set_xlim(-2,2); ax.set_ylim(-2,2); ax.set_zlim(-2,2)
    plt.draw()