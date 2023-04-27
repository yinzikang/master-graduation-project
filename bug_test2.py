from gym_custom.envs.controller import *
from gym_custom.envs.transformations import random_rotation_matrix, quaternion_from_matrix, \
    quaternion_multiply

# 姿态误差测试
for _ in range(10):
    mat1 = random_rotation_matrix()
    mat2 = random_rotation_matrix()
    quat1 = quaternion_from_matrix(mat1)
    quat2 = quaternion_from_matrix(mat2)
    # print(quat1)
    # print(quat2)
    o_a_m = orientation_error_axis_angle_with_mat(mat2[:3, :3], mat1[:3, :3])
    o_q_m = orientation_error_quat_with_mat(mat2[:3, :3], mat1[:3, :3])
    o_q_q = orientation_error_quat_with_quat(quat2, quat1)
    print(o_a_m)
    print(o_q_m)
    print(o_q_q)
    # if max(abs(o_q_m - o_q_q)) > 0.001:
    #     print(quat1)
    #     print(quat2)
    #     print(o_q_m)
    #     print(o_q_q)
    #     print(o_q_m - o_q_q)

# for _ in range(10):
#     mat1 = random_rotation_matrix()
#     mat2 = random_rotation_matrix()
#     quat1 = quaternion_from_matrix(mat1)
#     quat2 = quaternion_from_matrix(mat2)
#     # mat计算
#     mat44 = np.eye(4)
#     mat44[:3, :3] = np.linalg.inv(mat2[:3, :3]) @ mat1[:3, :3]
#     quat_from_mat = quaternion_from_matrix(mat44)
#     if quat_from_mat[3] < 0:
#         quat_from_mat = - quat_from_mat
#     o_q_m = mat2[:3, :3] @ quat_from_mat[:3]
#     # quat计算
#     quat_from_quat = quaternion_multiply(quaternion_inverse(quat2), quat1)
#     if quat_from_quat[3] < 0:
#         quat_from_quat = - quat_from_quat
#     quat_from_quat[3] = 0
#     o_q_q = quaternion_multiply(quaternion_multiply(quat2, quat_from_quat), quaternion_inverse(quat2))
#     o_q_q = o_q_q[:3]
#     # print(quat_from_mat)
#     # print(quat_from_quat)
#     print(o_q_m)