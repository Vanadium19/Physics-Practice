from rotation_quaternions import rotate

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


class Dumbbell:
    def __init__(self, l1: float, l2: float, m1: float, m2: float):
        self.l1 = l1
        self.l2 = l2
        self.m1 = m1
        self.m2 = m2

        # Главные моменты инерции из решения задачи №3.
        self.A = (m1 * l1 ** 2) / 6.0 + (m1 * l2 ** 2) / 2.0 + (m2 * l2 ** 2) / 12.0
        self.B = self.A
        self.C = (m1 * l1 ** 2) / 3.0

        half_side = l1 / 2.0
        half_handle = l2 / 2.0

        self.top_plate_local = np.array([
            [half_side, half_side, half_handle],
            [half_side, -half_side, half_handle],
            [-half_side, -half_side, half_handle],
            [-half_side, half_side, half_handle],
        ]).T
        self.bottom_plate_local = np.array([
            [half_side, half_side, -half_handle],
            [half_side, -half_side, -half_handle],
            [-half_side, -half_side, -half_handle],
            [-half_side, half_side, -half_handle],
        ]).T
        self.handle_local = np.array([
            [0.0, 0.0, -half_handle],
            [0.0, 0.0, half_handle],
        ]).T

        self.r_c = np.array([0.0, 0.0, 0.0])
        self.Q = np.array([1.0, 0.0, 0.0, 0.0])

    def get_world_points(self):
        return {
            'top_plate': rotate(self.Q, self.top_plate_local) + self.r_c.reshape(3, 1),
            'bottom_plate': rotate(self.Q, self.bottom_plate_local) + self.r_c.reshape(3, 1),
            'handle': rotate(self.Q, self.handle_local) + self.r_c.reshape(3, 1),
        }

    def get_symmetry_axis_tip(self):
        axis_tip_local = np.array([[0.0], [0.0], [self.l2 / 2.0]])
        return (rotate(self.Q, axis_tip_local) + self.r_c.reshape(3, 1)).reshape(3)


def _draw_square(square_points: np.ndarray, color: str, linewidth: float = 2.0):
    for index in range(4):
        next_index = (index + 1) % 4
        ax.plot(
            [square_points[0, index], square_points[0, next_index]],
            [square_points[1, index], square_points[1, next_index]],
            [square_points[2, index], square_points[2, next_index]],
            c=color,
            linewidth=linewidth,
        )


def draw_dumbbell(dumbbell: Dumbbell, symmetry_axis_trace: Optional[np.ndarray] = None):
    ax.clear()

    points = dumbbell.get_world_points()
    _draw_square(points['top_plate'], 'dimgray')
    _draw_square(points['bottom_plate'], 'dimgray')

    handle = points['handle']
    ax.plot(handle[0], handle[1], handle[2], c='firebrick', linewidth=3.0)
    ax.scatter([dumbbell.r_c[0]], [dumbbell.r_c[1]], [dumbbell.r_c[2]], c='royalblue', s=25)

    if symmetry_axis_trace is not None and len(symmetry_axis_trace) > 1:
        ax.plot(
            symmetry_axis_trace[:, 0],
            symmetry_axis_trace[:, 1],
            symmetry_axis_trace[:, 2],
            c='steelblue',
            linewidth=1.0,
            alpha=0.8,
        )

    radius = max(dumbbell.l2, dumbbell.l1) * 1.25
    ax.set_xlim(dumbbell.r_c[0] - radius, dumbbell.r_c[0] + radius)
    ax.set_ylim(dumbbell.r_c[1] - radius, dumbbell.r_c[1] + radius)
    ax.set_zlim(dumbbell.r_c[2] - radius, dumbbell.r_c[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.draw()
