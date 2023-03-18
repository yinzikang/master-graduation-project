#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Write target here

Write detailed description here

Write typical usage example here

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
1/10/23 7:37 PM   yinzikang      1.0         None
"""

# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""jk5杆系统环境，用于开门任务

除了运动学部分用kdl完成，其他均由mujoco实现

Write typical usage example here

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
11/01/23 2:53 PM   yinzikang      1.0         None
"""
import PyKDL as kdl
import numpy as np
import mujoco as mp

from utils.custom_logx import EpisodeLogger
from utils.custom_viewer import EnvViewer

import gym
import copy
from datetime import datetime
from spinup.utils.mpi_tools import proc_id


class Jk5StickRobotEnv:
    def __init__(self, mjc_model_path, task, qpos_init_list, p_bias, r_bias, step_num,
                 desired_posture_list, desired_vel_list, desired_acc_list, desired_force_list,
                 init_M, init_B, init_K, min_K, max_K):
        # robot part #########################################################################
        self.PI = np.pi
        self.joint_num = 6
        self.qpos_init_list = qpos_init_list
        self.task = task
        # 两个模型
        self.mjc_model = mp.MjModel.from_xml_path(filename=mjc_model_path)
        self.kdl_model = self.CreateKdlModel()
        self.p_bias = p_bias.copy()  # mujoco模型的位置偏置
        self.r_bias = r_bias.copy()  # mujoco模型的旋转偏置
        # 机器人各部位
        self.joint_list = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        self.actuator_list = ['motor1', 'motor2', 'motor3', 'motor4', 'motor5', 'motor6']
        self.sensor_list = ['contact_force', 'contact_torque', 'contact_touch', 'nft_force', 'nft_torque']
        self.eef_name = 'ee'
        self.eef_id = mp.mj_name2id(self.mjc_model, mp.mjtObj.mjOBJ_SITE, self.eef_name)

        # impedance control part,阻抗参数 #####################################################
        self.step_num = step_num
        self.desired_posture_list = desired_posture_list
        self.desired_vel_list = desired_vel_list.copy()
        self.desired_acc_list = desired_acc_list.copy()
        self.desired_force_list = desired_force_list.copy()

        self.init_M = init_M.copy()
        self.init_B = init_B.copy()
        self.init_K = init_K.copy()
        self.min_K = min_K.copy()
        self.max_K = max_K.copy()

        self.M = None
        self.B = None
        self.K = None

        self.last_tau = None
        self.last_jacobian = None

        # reinforcement learning part,每一部分有超参数与参数组成 ##################################
        # o, a, r
        self.observation_num = 6  # 观测数：3个位置+3个速度
        self.action_num = 3  # 动作数：接触方向的刚度变化量
        self.observation_space = gym.spaces.Box(low=-np.inf * np.ones(self.observation_num),
                                                high=np.inf * np.ones(self.observation_num),
                                                dtype=np.float32)  # 连续观测空间
        self.action_space = gym.spaces.Box(low=-1000 * np.ones(self.action_num),
                                           high=1000 * np.ones(self.action_num),
                                           dtype=np.float32)  # 连续动作空间
        self.current_step = 0
        self.current_episode = -1
        # reset
        self.data = mp.MjData(self.mjc_model)
        self.initial_state = copy.deepcopy(self.data)

        timestamp = "{0:%Y-%m-%d-%H-%M-%S/}".format(datetime.now())
        self.file_dir = './results/' + str(self.task) + '/' + timestamp
        # self.tensorboard_dir = './logs/' + str(self.task) + '/' + timestamp
        # self.writer = SummaryWriter(self.tensorboard_dir)
        # if not os.path.exists(self.file_dir):
        #     os.makedirs(self.file_dir)

    def CreateKdlModel(self):
        # 标准kdl
        rbt = kdl.Chain()
        rbt.addSegment(kdl.Segment("base_link", kdl.Joint("base_joint", kdl.Joint.Fixed),
                                   kdl.Frame(kdl.Rotation.Identity(), kdl.Vector(0, 0, 0.069))))
        rbt.addSegment(kdl.Segment("link1", kdl.Joint("joint1", kdl.Joint.RotZ),
                                   kdl.Frame(kdl.Rotation.RotX(np.pi / 2), kdl.Vector(0, 0, 0.073))))
        rbt.addSegment(kdl.Segment("link2", kdl.Joint("joint2", kdl.Joint.RotZ),
                                   kdl.Frame(kdl.Rotation.RotZ(np.pi / 2), kdl.Vector(0, 0.425, 0))))
        rbt.addSegment(kdl.Segment("link3", kdl.Joint("joint3", kdl.Joint.RotZ),
                                   kdl.Frame(kdl.Rotation.Identity(), kdl.Vector(0.395, 0, 0))))
        rbt.addSegment(kdl.Segment("link4", kdl.Joint("joint4", kdl.Joint.RotZ),
                                   kdl.Frame(kdl.Rotation.Quaternion(-0.5, 0.5, -0.5, 0.5), kdl.Vector(0, 0, 0.1135))))
        rbt.addSegment(kdl.Segment("link5", kdl.Joint("joint5", kdl.Joint.RotZ),
                                   kdl.Frame(kdl.Rotation.RotX(np.pi / 2), kdl.Vector(0, 0, 0.1015))))
        rbt.addSegment(kdl.Segment("link6", kdl.Joint("joint6", kdl.Joint.RotZ),
                                   kdl.Frame(kdl.Rotation.Identity(), kdl.Vector(0, 0, 0.094))))
        if self.task == 'desk':  # 桌任务ee长0.169
            rbt.addSegment(kdl.Segment("end-effector", kdl.Joint("ee_joint", kdl.Joint.Fixed),
                                       kdl.Frame(kdl.Rotation.Identity(), kdl.Vector(0, 0, 0.169))))
        if self.task == 'open door':  # 门任务ee长0.06
            rbt.addSegment(kdl.Segment("end-effector", kdl.Joint("ee_joint", kdl.Joint.Fixed),
                                       kdl.Frame(kdl.Rotation.Identity(), kdl.Vector(0, 0, 0.05))))
        if self.task == 'close door':  # 门任务ee长0.06
            rbt.addSegment(kdl.Segment("end-effector", kdl.Joint("ee_joint", kdl.Joint.Fixed),
                                       kdl.Frame(kdl.Rotation.Identity(), kdl.Vector(0, 0, 0.063))))

        return rbt

    def ToKdlQPos(self, qpos_list):
        kdl_qpos_list = kdl.JntArray(self.joint_num)
        for i in range(self.joint_num):
            kdl_qpos_list[i] = qpos_list[i]
        return kdl_qpos_list

    def ToKdlFrame(self, pos, mat):
        pos = pos.reshape(-1)
        mat = mat.reshape(-1)
        kdl_xpos = kdl.Vector(pos[0], pos[1], pos[2])
        kdl_xmat = kdl.Rotation(mat[0], mat[1], mat[2],
                                mat[3], mat[4], mat[5],
                                mat[6], mat[7], mat[8])
        kdl_frame = kdl.Frame(kdl_xmat, kdl_xpos)
        return kdl_frame

    def ToNumpyFrame(self, kdl_frame):
        kdl_pos = kdl_frame.p
        kdl_mat = kdl_frame.M
        pos = np.array([kdl_pos[0], kdl_pos[1], kdl_pos[2]])
        mat = np.array([kdl_mat[0, 0], kdl_mat[0, 1], kdl_mat[0, 2],
                        kdl_mat[1, 0], kdl_mat[1, 1], kdl_mat[1, 2],
                        kdl_mat[2, 0], kdl_mat[2, 1], kdl_mat[2, 2]]).reshape(3, 3)
        return pos, mat

    def GetXPos(self):
        xpos = np.array(self.data.site(self.eef_name).xpos)
        xmat = np.array(self.data.site(self.eef_name).xmat).reshape(3, 3)
        xvel = np.ndarray(shape=(6,), dtype=np.float64, order='C')
        mp.mj_objectVelocity(self.mjc_model, self.data, mp.mjtObj.mjOBJ_SITE, self.eef_id, xvel, False)
        xvelp, xvelr = xvel[3:], xvel[:3]
        return xpos, xmat, xvelp, xvelr

    def GetQPos(self):
        qpos = np.array([self.data.joint(joint_name).qpos for joint_name in self.joint_list]).reshape(-1)
        qvel = np.array([self.data.joint(joint_name).qvel for joint_name in self.joint_list]).reshape(-1)
        return qpos, qvel

    def ForwardKinematics(self, qpos_list):
        kdl_qpos_list = self.ToKdlQPos(qpos_list)
        kdl_frame = kdl.Frame()
        fksolver = kdl.ChainFkSolverPos_recursive(self.kdl_model)
        fksolver.JntToCart(kdl_qpos_list, kdl_frame)
        pos, mat = self.ToNumpyFrame(kdl_frame)
        pos = pos.reshape(3, 1)
        mat = mat.reshape(3, 3)
        pos = self.r_bias @ pos + self.p_bias.reshape(3, 1)
        mat = self.r_bias @ mat
        return pos.reshape(-1), mat

    def InverseKinematics(self, qpos_init_list, xpos, xmat):
        xpos = xpos.reshape(3, 1)
        xmat = xmat.reshape(3, 3)
        xpos = np.linalg.inv(self.r_bias) @ (xpos - self.p_bias.reshape(3, 1))
        xmat = np.linalg.inv(self.r_bias) @ xmat
        kdl_qpos_init_list = self.ToKdlQPos(qpos_init_list)
        kdl_frame = self.ToKdlFrame(xpos, xmat)
        # iksolver = kdl.ChainIk,SolverPos_LMA(self.kdl_model, maxiter=20000)
        # kdl_joint_pos = kdl.JntArray(self.joint_num)
        # iksolver.CartToJnt(kdl_qpos_init_list, kdl_frame, kdl_joint_pos)
        # return kdl_joint_pos

        kdl_joint_pos_min = kdl.JntArray(self.joint_num)
        kdl_joint_pos_max = kdl.JntArray(self.joint_num)
        for joint_id in range(self.joint_num):
            kdl_joint_pos_min[joint_id] = -self.PI
            kdl_joint_pos_max[joint_id] = self.PI
        fkpossolver = kdl.ChainFkSolverPos_recursive(self.kdl_model)
        ikvelsolver = kdl.ChainIkSolverVel_pinv(self.kdl_model)
        ik = kdl.ChainIkSolverPos_NR_JL(self.kdl_model, kdl_joint_pos_min, kdl_joint_pos_max,
                                        fkpossolver, ikvelsolver, maxiter=2000)
        kdl_joint_pos = kdl.JntArray(self.joint_num)
        ik.CartToJnt(kdl_qpos_init_list, kdl_frame, kdl_joint_pos)
        return kdl_joint_pos

    def orientation_error(self, desired, current):
        rc1 = current[0:3, 0]
        rc2 = current[0:3, 1]
        rc3 = current[0:3, 2]
        rd1 = desired[0:3, 0]
        rd2 = desired[0:3, 1]
        rd3 = desired[0:3, 2]
        error = 0.5 * (np.cross(rc1, rd1) + np.cross(rc2, rd2) + np.cross(rc3, rd3))
        return error

    def TrajectoryGeneration(self, cart_init_pos, cart_end_pos, dot_num):
        trajectory = []
        for i in range(dot_num):
            trajectory.append(cart_init_pos + (cart_end_pos - cart_init_pos) / (dot_num - 1) * i)
        return trajectory

    def GetMassMatrix(self):  # 注意行列数与具体关节数不一定相等，且所需行列不一定相邻
        mass_matrix = np.ndarray(shape=(self.mjc_model.nv, self.mjc_model.nv), dtype=np.float64, order='C')
        mp.mj_fullM(self.mjc_model, mass_matrix, self.data.qM)
        mass_matrix = mass_matrix[:self.joint_num, :self.joint_num]
        return mass_matrix

    def GetJacobian(self):  # 注意行列数与具体关节数不一定相等，且所需行列不一定相邻
        jacp = np.ndarray(shape=(3, self.mjc_model.nv), dtype=np.float64, order='C')
        jacr = np.ndarray(shape=(3, self.mjc_model.nv), dtype=np.float64, order='C')
        mp.mj_jacSite(self.mjc_model, self.data, jacp, jacr, self.eef_id)
        J_full = np.array(np.vstack([jacp[:, :self.joint_num], jacr[:, :self.joint_num]]))
        return J_full

    def GetJacobianDot(self):
        cur_jacobian = self.GetJacobian()
        jacobian_d = np.subtract(cur_jacobian, self.last_jacobian)
        jacobian_d = jacobian_d / self.mjc_model.opt.timestep
        self.last_jacobian = cur_jacobian
        return jacobian_d

    def GetObservation(self):
        pos, mat, v, w = self.GetXPos()
        q, qd = self.GetQPos()
        observation = np.concatenate((pos, v))
        other_info = (pos, mat, v, w, q, qd)
        return observation, other_info  # (6,)

    def GetReward(self, observation, action, x_error, xd_error, contact_force, nft_force, tau):
        if self.task == 'desk':
            # movement_reward = - 100 * np.sqrt(np.sum(x_error[[0, 3, 4, 5]] ** 2))
            movement_reward = - 100 * np.sum(x_error[[0, 1, 3, 4, 5]] ** 2)
            # fext_reward = 10 - np.sqrt(np.sum((contact_force - self.desired_force_list[step]) ** 2))
            fext_reward = 20 - np.sum(np.linalg.norm(contact_force - self.desired_force_list[self.current_step], ord=1))
            tau_reward = - np.sqrt(np.sum((tau - self.last_tau) ** 2))
            return fext_reward
        if self.task == 'open door':
            movement_reward = - 10 * np.sqrt(np.sum(x_error ** 2))
            tau_reward = - np.sqrt(np.sum((tau - self.last_tau) ** 2))
            return movement_reward
        if self.task == 'close door':
            movement_reward = - 10 * np.sqrt(np.sum(x_error ** 2))
            tau_reward = - np.sqrt(np.sum((tau - self.last_tau) ** 2))
            return movement_reward

    def reset(self):
        """
        重置环境，返回观测
        :return:
        """
        # robot reset #######################################################################
        self.data = copy.deepcopy(self.initial_state)
        for i in range(self.joint_num):
            self.data.joint(self.joint_list[i]).qpos = self.qpos_init_list[i]
        mp.mj_forward(self.mjc_model, self.data)
        if self.task == "open door":
            pos1 = np.array(self.data.body("dummy_body").xpos)
            pos2 = np.array(self.data.body("anchor").xpos)
            mat1 = np.array(self.data.body("dummy_body").xmat).reshape(3, 3)
            pos_delta1 = np.matmul(np.linalg.inv(mat1), (pos2 - pos1).reshape(3, 1)).reshape(-1)
            self.mjc_model.equality('test').data = np.hstack((pos_delta1, np.zeros(4)))
            self.mjc_model.equality('test').active = True

            pos1 = np.array(self.data.body("dummy_body").xpos)
            pos2 = np.array(self.data.body("door_board").xpos)
            mat1 = np.array(self.data.body("dummy_body").xmat).reshape(3, 3)
            pos_delta2 = np.matmul(np.linalg.inv(mat1), (pos2 - pos1).reshape(3, 1)).reshape(-1)
            self.mjc_model.equality('robot2door').data = np.hstack((pos_delta2, np.zeros(4)))
            self.mjc_model.equality('robot2door').active = True

        if self.task == "close door":
            self.data.joint('hinge').qpos = np.pi / 2

        mp.mj_forward(self.mjc_model, self.data)
        self.last_tau = np.array(self.data.qfrc_bias[:self.joint_num])
        self.last_jacobian = self.GetJacobian()

        # impedance control reset
        self.M = self.init_M.copy()
        self.B = self.init_B.copy()
        self.K = self.init_K.copy()

        # algorithm reset ##################################################################
        # 次数更新
        self.current_step = 0
        self.current_episode += 1

        observation, other_info = self.GetObservation()
        return observation

    def step(self, action):
        """
        运行动作，获得下一次观测
        :param action:
        :return:
        """
        # 获取观测
        observation, (x_pos, x_mat, x_pos_vel, x_mat_vel, q, qd) = self.GetObservation()
        # self.K[:3] += action  # 三个方向的
        # self.K[2:3] += action[2]  # 只有z方向
        self.K[2:3] = action[2] + 1100  # 用于防止约束
        # np.clip(self.K, self.min_K, self.max_K,out=self.K)

        desired_pos = self.desired_posture_list[self.current_step][:3]
        desired_mat = self.desired_posture_list[self.current_step][3:].reshape(3, 3)
        x_error = np.concatenate([desired_pos - x_pos, self.orientation_error(desired_mat, x_mat)])
        xd_error = self.desired_vel_list[self.current_step] - np.concatenate([x_pos_vel, x_mat_vel])
        contact_force = np.concatenate((self.data.sensor(self.sensor_list[0]).data,
                                        self.data.sensor(self.sensor_list[1]).data))
        touch_force = np.array(self.data.sensor(self.sensor_list[2]).data)
        nft_force = np.concatenate((self.data.sensor(self.sensor_list[3]).data,
                                    self.data.sensor(self.sensor_list[4]).data))
        cg = np.array(self.data.qfrc_bias[:self.joint_num])

        # 阻抗力矩计算
        D_q = self.GetMassMatrix()
        J = self.GetJacobian()
        J_inv = np.linalg.inv(J)
        J_T_inv = np.linalg.inv(J.T)
        Jd = self.GetJacobianDot()
        D_x = np.dot(J_T_inv, np.dot(D_q, J_inv))
        h_x = np.dot(J_T_inv, cg) - np.dot(np.dot(D_x, Jd), qd)
        # T = np.multiply(self.B, xd_error) + np.multiply(self.K, x_error)
        T = np.multiply(self.B, xd_error) + np.multiply(self.K, x_error) + contact_force
        # T = np.multiply(self.B, xd_error) + np.multiply(self.K, x_error) - self.desired_force_list[step] + contact_force
        V = self.desired_acc_list[self.current_step] + np.dot(np.linalg.inv(self.M), T)
        F = np.dot(D_x, V) + h_x - contact_force
        tau = np.dot(J.T, F)
        # print(contact_force[3:])
        # print(contact_force[:3], touch_force, np.linalg.norm(contact_force[:3]) - touch_force)
        # print(desired_pos[2], x_pos[2], x_error[2], contact_force[2] / x_error[2])

        # 执行
        self.data.ctrl[:] = tau
        mp.mj_step2(self.mjc_model, self.data)
        mp.mj_step1(self.mjc_model, self.data)
        # mp.mj_step(self.mjc_model, self.data)
        # mp.mj_forward(self.mjc_model, self.data)
        self.last_tau = tau.copy()
        # 执行完action计算该(observation,action)获得的奖励
        next_observation, (x_pos, x_mat, x_pos_vel, x_mat_vel, q, qd) = self.GetObservation()
        reward = self.GetReward(observation, action,
                                x_error, xd_error,
                                contact_force, nft_force, tau)

        # 可视化
        if hasattr(self, 'viewer'):
            self.viewer.render()
        if hasattr(self, 'logger'):
            self.logger.store_buffer(K_series=self.K.copy(),
                                     contact_force_series=contact_force,
                                     nft_force_series=nft_force,
                                     tau_series=tau,
                                     q_series=q,
                                     qd_series=qd,

                                     observation_series=observation,
                                     action_series=action,
                                     reward_series=reward)

        dead = True if self.current_step + 1 == self.step_num else False
        other_info = None
        self.current_step += 1

        return next_observation, reward, dead, other_info

    def logger_init(self):
        if proc_id() == 0:
            self.logger = EpisodeLogger()

    def viewer_init(self):
        if proc_id() == 0:
            self.viewer = EnvViewer(self)
