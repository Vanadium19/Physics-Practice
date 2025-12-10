from scipy.spatial.transform import Rotation as R

from rotation_matrixes import *

# Эйлер
A_zxz = R.from_euler("ZXZ", [125, 0, 85], degrees=True)
A1 = A_zxz.as_matrix()
A2 = A_z(50) @ A_x(0) @ A_z(160)
print("Match: ", np.allclose(A1, A2))

# Самолётные
A_zxy = R.from_euler("ZXY", [0, -90, 180], degrees=True)
A3 = A_zxy.as_matrix()
A4 = A_z(190) @ A_x(-90) @ A_y(10)
print("Match: ", np.allclose(A3, A4))