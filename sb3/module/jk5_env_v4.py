#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""基于gym.Env的实现，用于sb3环境

继承自v3,整合力矩控制以及轨迹规划
将旋转表示为四元数

Write typical usage example here

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
3/2/23 11:08 AM   yinzikang      1.0         None
"""
import sys

import numpy

sys.path.append('..')
import PyKDL as kdl
import numpy as np
import mujoco as mp
import matplotlib.pyplot as plt

from utils.custom_logx import EpisodeLogger
from utils.custom_viewer import EnvViewer
from module.transformations import quaternion_from_matrix, quaternion_matrix, quaternion_slerp

from gym import spaces, Env
import copy
from spinup.utils.mpi_tools import proc_id
from stable_baselines3.common.env_checker import check_env


class Jk5StickEnv:

    def __init__(self, mjc_model_path, task, qpos_init_list, p_bias, r_bias):
        # robot part #########################################################################
        self.joint_num = 6
        self.qpos_init_list = qpos_init_list
        self.task = task
        # 两个模型
        self.mjc_model = mp.MjModel.from_xml_path(filename=mjc_model_path)
        self.kdl_model = self.create_kdl_model()
        self.control_frequency = int(1 / self.mjc_model.opt.timestep)  # 控制频率
        self.p_bias = p_bias.copy()  # mujoco模型的位置偏置
        self.r_bias = r_bias.copy()  # mujoco模型的旋转偏置
        # 机器人各部位
        self.joint_list = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        self.actuator_list = ['motor1', 'motor2', 'motor3', 'motor4', 'motor5', 'motor6']
        self.sensor_list = ['contact_force', 'contact_torque', 'contact_touch', 'nft_force', 'nft_torque']
        self.eef_name = 'ee'
        self.eef_id = mp.mj_name2id(self.mjc_model, mp.mjtObj.mjOBJ_SITE, self.eef_name)
        # data
        self.data = mp.MjData(self.mjc_model)

        self.last_jacobian = None  # 用于计算jd

    def create_kdl_model(self):
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

    def to_kdl_qpos(self, qpos_list):
        kdl_qpos_list = kdl.JntArray(self.joint_num)
        for i in range(self.joint_num):
            kdl_qpos_list[i] = qpos_list[i]
        return kdl_qpos_list

    @staticmethod
    def to_kdl_frame(pos, mat):
        pos = pos.reshape(-1)
        mat = mat.reshape(-1)
        kdl_xpos = kdl.Vector(pos[0], pos[1], pos[2])
        kdl_xmat = kdl.Rotation(mat[0], mat[1], mat[2],
                                mat[3], mat[4], mat[5],
                                mat[6], mat[7], mat[8])
        kdl_frame = kdl.Frame(kdl_xmat, kdl_xpos)
        return kdl_frame

    @staticmethod
    def to_numpy_frame(kdl_frame):
        kdl_pos = kdl_frame.p
        kdl_mat = kdl_frame.M
        pos = np.array([kdl_pos[0], kdl_pos[1], kdl_pos[2]])
        mat = np.array([kdl_mat[0, 0], kdl_mat[0, 1], kdl_mat[0, 2],
                        kdl_mat[1, 0], kdl_mat[1, 1], kdl_mat[1, 2],
                        kdl_mat[2, 0], kdl_mat[2, 1], kdl_mat[2, 2]]).reshape(3, 3)
        return pos, mat

    def get_xpos_xvel(self):
        xpos = np.array(self.data.site(self.eef_name).xpos)
        xmat = np.array(self.data.site(self.eef_name).xmat).reshape(3, 3)
        xvel = np.ndarray(shape=(6,), dtype=np.float64, order='C')
        mp.mj_objectVelocity(self.mjc_model, self.data, mp.mjtObj.mjOBJ_SITE, self.eef_id, xvel, False)
        xvelp, xvelr = xvel[3:], xvel[:3]
        return xpos, xmat, xvelp, xvelr

    def get_qpos_qvel(self):
        qpos = np.array([self.data.joint(joint_name).qpos for joint_name in self.joint_list]).reshape(-1)
        qvel = np.array([self.data.joint(joint_name).qvel for joint_name in self.joint_list]).reshape(-1)
        return qpos, qvel

    def forward_kinematics(self, qpos_list):
        kdl_qpos_list = self.to_kdl_qpos(qpos_list)
        kdl_frame = kdl.Frame()
        fksolver = kdl.ChainFkSolverPos_recursive(self.kdl_model)
        fksolver.JntToCart(kdl_qpos_list, kdl_frame)
        pos, mat = self.to_numpy_frame(kdl_frame)
        pos = pos.reshape(3, 1)
        mat = mat.reshape(3, 3)
        pos = self.r_bias @ pos + self.p_bias.reshape(3, 1)
        mat = self.r_bias @ mat
        return pos.reshape(-1), mat

    def inverse_kinematics(self, qpos_init_list, xpos, xmat):
        xpos = xpos.reshape(3, 1)
        xmat = xmat.reshape(3, 3)
        xpos = np.linalg.inv(self.r_bias) @ (xpos - self.p_bias.reshape(3, 1))
        xmat = np.linalg.inv(self.r_bias) @ xmat
        kdl_qpos_init_list = self.to_kdl_qpos(qpos_init_list)
        kdl_frame = self.to_kdl_frame(xpos, xmat)
        # iksolver = kdl.ChainIk,SolverPos_LMA(self.kdl_model, maxiter=20000)
        # kdl_joint_pos = kdl.JntArray(self.joint_num)
        # iksolver.CartToJnt(kdl_qpos_init_list, kdl_frame, kdl_joint_pos)
        # return kdl_joint_pos

        kdl_joint_pos_min = kdl.JntArray(self.joint_num)
        kdl_joint_pos_max = kdl.JntArray(self.joint_num)
        for joint_id in range(self.joint_num):
            kdl_joint_pos_min[joint_id] = -np.pi
            kdl_joint_pos_max[joint_id] = np.pi
        fkpossolver = kdl.ChainFkSolverPos_recursive(self.kdl_model)
        ikvelsolver = kdl.ChainIkSolverVel_pinv(self.kdl_model)
        ik = kdl.ChainIkSolverPos_NR_JL(self.kdl_model, kdl_joint_pos_min, kdl_joint_pos_max,
                                        fkpossolver, ikvelsolver, maxiter=2000)
        kdl_joint_pos = kdl.JntArray(self.joint_num)
        ik.CartToJnt(kdl_qpos_init_list, kdl_frame, kdl_joint_pos)
        return kdl_joint_pos

    @staticmethod
    def orientation_error(desired, current):
        rc1 = current[0:3, 0]
        rc2 = current[0:3, 1]
        rc3 = current[0:3, 2]
        rd1 = desired[0:3, 0]
        rd2 = desired[0:3, 1]
        rd3 = desired[0:3, 2]
        error = 0.5 * (np.cross(rc1, rd1) + np.cross(rc2, rd2) + np.cross(rc3, rd3))
        return error

    @staticmethod
    def generate_trajectory(cart_init_pos, cart_end_pos, dot_num):
        trajectory = []
        for i in range(dot_num):
            trajectory.append(cart_init_pos + (cart_end_pos - cart_init_pos) / (dot_num - 1) * i)
        return trajectory

    def get_mass_matrix(self):  # 注意行列数与具体关节数不一定相等，且所需行列不一定相邻
        mass_matrix = np.ndarray(shape=(self.mjc_model.nv, self.mjc_model.nv), dtype=np.float64, order='C')
        mp.mj_fullM(self.mjc_model, mass_matrix, self.data.qM)
        mass_matrix = mass_matrix[:self.joint_num, :self.joint_num]
        return mass_matrix

    def get_jacobian(self):  # 注意行列数与具体关节数不一定相等，且所需行列不一定相邻
        jacp = np.ndarray(shape=(3, self.mjc_model.nv), dtype=np.float64, order='C')
        jacr = np.ndarray(shape=(3, self.mjc_model.nv), dtype=np.float64, order='C')
        mp.mj_jacSite(self.mjc_model, self.data, jacp, jacr, self.eef_id)
        J_full = np.array(np.vstack([jacp[:, :self.joint_num], jacr[:, :self.joint_num]]))
        return J_full

    def get_jacobian_dot(self):
        cur_jacobian = self.get_jacobian()
        jacobian_d = np.subtract(cur_jacobian, self.last_jacobian)
        jacobian_d = jacobian_d / self.mjc_model.opt.timestep
        self.last_jacobian = cur_jacobian
        return jacobian_d


class Jk5StickStiffnessEnv(Jk5StickEnv):
    def __init__(self, mjc_model_path, task, qpos_init_list, p_bias, r_bias, step_num,
                 desired_posture_list, desired_vel_list, desired_acc_list, desired_force_list,
                 init_M, init_B, init_K, min_K, max_K):
        super().__init__(mjc_model_path, task, qpos_init_list, p_bias, r_bias)
        self.initial_state = copy.deepcopy(self.data)
        self.step_num = step_num
        # impedance control part,阻抗参数 #####################################################
        self.desired_posture_list = desired_posture_list.copy()
        self.desired_vel_list = desired_vel_list.copy()
        self.desired_acc_list = desired_acc_list.copy()
        self.desired_force_list = desired_force_list.copy()

        self.init_M = init_M.copy()
        self.init_B = init_B.copy()
        self.init_K = init_K.copy()
        self.min_K = min_K.copy()
        self.max_K = max_K.copy()
        self.wn = np.sqrt(self.init_K / np.diagonal(self.init_M))
        self.damping_ratio = self.init_B / 2 / np.sqrt(self.init_K * np.diagonal(self.init_M))

        self.M = None
        self.B = None
        self.K = None

        self.current_step = 0

    def get_status(self):
        xpos, xmat, xpos_vel, xmat_vel = self.get_xpos_xvel()
        mat44 = np.eye(4)
        mat44[:3, :3] = xmat
        quat = quaternion_from_matrix(mat44)
        qpos, qvel = self.get_qpos_qvel()

        desired_pos = self.desired_posture_list[self.current_step][:3]
        desired_mat = self.desired_posture_list[self.current_step][3:].reshape(3, 3)
        xpos_error = np.concatenate([desired_pos - xpos, self.orientation_error(desired_mat, xmat)])
        xvel_error = self.desired_vel_list[self.current_step] - np.concatenate([xpos_vel, xmat_vel])

        contact_force = np.concatenate((self.data.sensor(self.sensor_list[0]).data,
                                        self.data.sensor(self.sensor_list[1]).data))
        touch_force = np.array(self.data.sensor(self.sensor_list[2]).data)
        nft_force = np.concatenate((self.data.sensor(self.sensor_list[3]).data,
                                    self.data.sensor(self.sensor_list[4]).data))

        tau = np.array(self.data.ctrl[:])

        status_dict = {'xpos': xpos,
                       'xmat': xmat,
                       'quat': quat,
                       'xpos_vel': xpos_vel,
                       'xmat_vel': xmat_vel,
                       'qpos': qpos,
                       'qvel': qvel,
                       'desired_pos': desired_pos,
                       'desired_mat': desired_mat,
                       'xpos_error': xpos_error,
                       'xvel_error': xvel_error,
                       'contact_force': contact_force,
                       'touch_force': touch_force,
                       'nft_force': nft_force,
                       'tau': tau}

        return status_dict

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

        self.last_jacobian = self.get_jacobian()

        # impedance control reset
        self.M = self.init_M.copy()
        self.B = self.init_B.copy()
        self.K = self.init_K.copy()

        # algorithm reset ##################################################################
        # 步数更新
        self.current_step = 0

        return self.get_status()

    def step(self, status_dict):
        """
        阻抗控制器运行一次
        """
        # 状态
        xpos_error = status_dict['xpos_error']
        xvel_error = status_dict['xvel_error']
        qpos = status_dict['qpos']
        qvel = status_dict['qvel']
        contact_force = status_dict['contact_force']
        nft_force = status_dict['nft_force']
        # 阻抗控制率
        cg = np.array(self.data.qfrc_bias[:self.joint_num])
        D_q = self.get_mass_matrix()
        J = self.get_jacobian()
        J_inv = np.linalg.inv(J)
        J_T_inv = np.linalg.inv(J.T)
        Jd = self.get_jacobian_dot()
        D_x = np.dot(J_T_inv, np.dot(D_q, J_inv))
        h_x = np.dot(J_T_inv, cg) - np.dot(np.dot(D_x, Jd), qvel)
        T = np.multiply(self.B, xvel_error) + np.multiply(self.K, xpos_error) + contact_force
        V = self.desired_acc_list[self.current_step] + np.dot(np.linalg.inv(self.M), T)
        F = np.dot(D_x, V) + h_x - contact_force
        tau = np.dot(J.T, F)
        # 导纳控制率

        # 执行
        self.data.ctrl[:] = tau
        mp.mj_step2(self.mjc_model, self.data)
        mp.mj_step1(self.mjc_model, self.data)

        # 可视化
        if hasattr(self, 'viewer'):
            self.viewer.render()
        if hasattr(self, 'logger'):
            self.logger.store_buffer(K_series=self.K.copy(),  # copy不能去，不然全是一个至
                                     contact_force_series=contact_force,
                                     nft_force_series=nft_force,
                                     tau_series=tau,
                                     q_series=qpos,
                                     qd_series=qvel)
        status_dict = self.get_status()
        self.current_step += 1
        return status_dict

    def logger_init(self, output_dir=None):
        if proc_id() == 0:
            self.logger = EpisodeLogger(output_dir)

    def viewer_init(self, pause_start=False):
        if proc_id() == 0:
            self.viewer = EnvViewer(self, pause_start)

    def render(self, mode="human", pause_start=False):
        if not hasattr(self, 'viewer'):
            self.viewer_init(pause_start)
        self.viewer.render()


class TrainEnv(Jk5StickStiffnessEnv, Env):

    def __init__(self, mjc_model_path, task, qpos_init_list, p_bias, r_bias, step_num, rl_frequency, observation_range,
                 desired_posture_list, desired_vel_list, desired_acc_list, desired_force_list,
                 init_M, init_B, init_K, min_K, max_K):

        super().__init__(mjc_model_path, task, qpos_init_list, p_bias, r_bias, step_num,
                         desired_posture_list, desired_vel_list, desired_acc_list, desired_force_list,
                         init_M, init_B, init_K, min_K, max_K)

        # reinforcement learning part,每一部分有超参数与参数组成 ##################################
        self.rl_frequency = rl_frequency
        self.sub_step_num = int(self.control_frequency / self.rl_frequency)  # 两个动作之间的机器人控制次数
        self.observation_range = observation_range  # 前observation_range个状态组成观测
        # buffer：用于计算observation
        self.observation_buffer = []
        # o, a, r，根据gym要求，设置为类变量，而不是实例变量
        self.observation_num = 10  # 观测数：3个位置+3个速度+4个旋转
        self.action_num = 6  # 动作数：接触方向的刚度变化量
        self.observation_space = spaces.Box(low=-np.inf * np.ones(self.observation_num),
                                            high=np.inf * np.ones(self.observation_num), dtype=np.float32)  # 连续观测空间
        self.action_space = spaces.Box(low=-1 * np.ones(self.action_num),
                                       high=1 * np.ones(self.action_num), dtype=np.float32)  # 连续动作空间
        self.action_limit = 50

        self.current_episode = -1

    def get_observation(self, status_dict):
        xpos = status_dict['xpos']
        quat = status_dict['quat']
        xpos_vel = status_dict['xpos_vel']

        status = np.concatenate((xpos, quat, xpos_vel), dtype=np.float32)
        self.observation_buffer.append(status)  # status与observation关系
        # 够长则取最近的observation_range个，不够对最远那个进行复制，observation_range×(3+4+2)
        if len(self.observation_buffer) >= self.observation_range:
            observation = np.array(self.observation_buffer[-self.observation_range:]).flatten()
        else:
            observation = np.array(self.observation_buffer +
                                   self.observation_buffer[0] * (
                                           self.current_step - self.observation_range)).flatten()
        return observation

    def reset(self):
        status_dict = super().reset()
        self.current_episode += 1
        self.observation_buffer = []

        return self.get_observation(status_dict)

    def step(self, action):
        """
        运行动作，获得下一次观测
        :param action:
        :return:
        """
        # 其余状态的初始化
        reward = 0
        done = False
        other_info = dict()
        # 对多次执行机器人控制
        status_dict = self.get_status()
        sub_step = 0
        for sub_step in range(self.sub_step_num):
            # action，即刚度变化量，进行插值
            self.K += self.action_limit * action / self.sub_step_num  # 只有z方向
            M = self.K / (self.wn * self.wn)
            self.B = 2 * self.damping_ratio * numpy.sqrt(M * self.K)
            self.M = np.diag(M)
            status_dict = super().step(status_dict)

            # 刚度越界或者任务结束，视为done
            if self.current_step + 1 == self.step_num:
                success = True
                other_info['is_success'], other_info["TimeLimit.truncated"], other_info[
                    'terminal info'] = True, True, 'success'
            else:
                success = False
            if any(np.greater(self.K, self.max_K)) or any(np.greater(self.min_K, self.K)):
                failure = True
                other_info['is_success'], other_info["TimeLimit.truncated"], other_info[
                    'terminal info'] = False, False, 'error K'
            else:
                failure = False
            done = success or failure

            # 获得奖励
            xpos_error = status_dict['xpos_error']
            contact_force = status_dict['contact_force']
            tau = status_dict['tau']
            ## 运动状态的奖励
            movement_reward = np.sum(xpos_error[[0, 1, 3, 4, 5]] ** 2)
            fext_reward = - np.sum(
                np.linalg.norm(contact_force - self.desired_force_list[self.current_step], ord=1))
            fext_reward = fext_reward + 10 if fext_reward > -5 else fext_reward  # 要是力距离期望力较近则进行额外奖励
            tau_reward = - np.sqrt(np.sum(tau ** 2))
            ## 任务结束的奖励与惩罚
            early_stop_penalty = (self.step_num - self.current_step) / self.step_num if done else 0
            error_k_penalty = -1 if failure else 0
            ## 接触力的奖励与惩罚
            zero_force_penalty = -1 if np.max(np.abs(contact_force)) < 1 else 0
            massive_force_penalty = -1 if np.max(np.abs(contact_force)) > 50 else 0

            # reward += 0 * movement_reward + 0.05 * fext_reward + 0 * tau_reward + \
            #           1 * error_k_penalty + 1 * early_stop_penalty + \
            #           0.5 * zero_force_penalty + 0 * massive_force_penalty
            reward += 0 * movement_reward + 0.05 * fext_reward + 0 * tau_reward + \
                      0 * error_k_penalty + 0 * early_stop_penalty + \
                      0 * zero_force_penalty + 0 * massive_force_penalty

            if hasattr(self, 'logger'):
                self.logger.store_buffer(action_series=action, reward_series=reward)

            if done:
                break

        return self.get_observation(status_dict), reward / (sub_step + 1), done, other_info

    def close(self):
        """Override close in your subclass to perform any necessary cleanup.

        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        if hasattr(self, 'viewer'):
            self.viewer.close()
            del self.viewer

    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).

        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.

        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        """
        return

    @property
    def unwrapped(self):
        """Completely unwrap this env.

        Returns:
            gym.Env: The base non-wrapped gym.Env instance
        """
        return self

    def __str__(self):
        if self.spec is None:
            return "<{} instance>".format(type(self).__name__)
        else:
            return "<{}<{}>>".format(type(self).__name__, self.spec.id)

    def __enter__(self):
        """Support with-statement for the environment."""
        return self

    def __exit__(self, *args):
        """Support with-statement for the environment."""
        self.close()
        # propagate exception
        return False


def load_env_kwargs(task=None):
    if task == 'desk':
        # 实验内容
        mjc_model_path = 'robot/jk5_table.xml'
        qpos_init_list = np.array([0, -30, 120, 0, -90, 0]) / 180 * np.pi
        p_bias = np.zeros(3)
        r_bias = np.eye(3)
        rl_frequency = 20  # 500可以整除，越大越多
        observation_range = 1
        step_num = 2000
        # 期望轨迹
        desired_pos_list = np.concatenate((np.linspace(-0.45, -0.75, step_num).reshape(step_num, 1),
                                           -0.1135 * np.ones((step_num, 1), dtype=float),
                                           0.05 * np.ones((step_num, 1), dtype=float)), axis=1)
        desired_mat_list = np.array([[0, -1, 0, -1, 0, 0, 0, 0, -1]], dtype=np.float64).repeat(step_num, axis=0)
        desired_posture_list = np.concatenate((desired_pos_list, desired_mat_list), axis=1)
        desired_vel_list = np.array([[-0.3 / step_num, 0, 0, 0, 0, 0]], dtype=np.float64).repeat(step_num, axis=0)
        desired_acc_list = np.array([[0, 0, 0, 0, 0, 0]], dtype=np.float64).repeat(step_num, axis=0)
        desired_force_list = np.array([[0, 0, 30, 0, 0, 0]], dtype=np.float64).repeat(step_num, axis=0)
        # 阻抗参数
        wn = 20
        damping_ratio = np.sqrt(2)
        init_K = np.array([1000, 1000, 1000, 1000, 1000, 1000], dtype=np.float64)
        M = init_K / (wn * wn)
        init_M = np.diag(M)
        init_B = 2 * damping_ratio * np.sqrt(M * init_K)
        min_K = np.array([100, 100, 100, 100, 100, 100], dtype=np.float64)
        max_K = np.array([3000, 3000, 3000, 3000, 3000, 3000], dtype=np.float64)
        # 用于Jk5StickStiffnessEnv的超参数
        rbt_env_kwargs = dict(mjc_model_path=mjc_model_path, task=task, qpos_init_list=qpos_init_list,
                              p_bias=p_bias, r_bias=r_bias,
                              step_num=step_num,
                              desired_posture_list=desired_posture_list, desired_vel_list=desired_vel_list,
                              desired_acc_list=desired_acc_list, desired_force_list=desired_force_list,
                              init_M=init_M, init_B=init_B, init_K=init_K, min_K=min_K,
                              max_K=max_K)
        # 用于TrainEnv的超参数
        rl_env_kwargs = copy.deepcopy(rbt_env_kwargs)
        rl_env_kwargs['rl_frequency'] = rl_frequency
        rl_env_kwargs['observation_range'] = observation_range

        return rbt_env_kwargs, rl_env_kwargs
    elif task == 'open door':
        # 实验内容
        mjc_model_path = 'robot/jk5_opendoor.xml'
        xpos_init_list = np.array([0.003, -0.81164083, 0.275])
        qpos_init_list = np.array([0, 60, 60, -30, -90, 0]) / 180 * np.pi
        p_bias = np.array([0, 0, 0.3885])
        r_bias = quaternion_matrix([0.5, 0.5, 0.5, 0.5])[:3, :3]
        rl_frequency = 250
        observation_range = 1
        step_num = 2000
        # 期望轨迹
        center_pred = np.array([-0.03, -0.4, 0.275])
        handle_pred = np.abs(xpos_init_list[0] - center_pred[0])  # 把手长度预测
        door_pred = np.abs(xpos_init_list[1] - center_pred[1])  # 门板长度预测
        radius_pred = np.linalg.norm(xpos_init_list[:2] - center_pred[:2])  # 半径预测
        angle_pred = np.arctan(handle_pred / door_pred)  # 角度预测
        total_angle = np.pi / 2  # 开门角度
        rot_matrix_init = np.eye(4)
        rot_matrix_init[:3, :3] = np.array([[0, 0, -1], [0, -1, 0], [-1, 0, 0]])  # 初始旋转矩阵
        quat_init = quaternion_from_matrix(rot_matrix_init)  # 初始四元数
        rot_matrix_last = np.eye(4)
        rot_matrix_last[:3, :3] = np.array([[0, 1, 0], [0, 0, -1], [-1, 0, -0]])  # 最终旋转矩阵
        quat_last = quaternion_from_matrix(rot_matrix_last)  # 最终四元数
        # 轨迹规划
        desired_pos_list = np.empty((step_num, 3))  # xpos traj
        desired_mat_list = np.empty((step_num, 9))  # xmat traj
        for i in range(step_num):
            desired_pos_list[i][0] = center_pred[0] + \
                                     radius_pred * np.sin(angle_pred + total_angle * i / (step_num - 1))
            desired_pos_list[i][1] = center_pred[1] \
                                     - radius_pred * np.cos(angle_pred + total_angle * i / (step_num - 1))
            desired_pos_list[i][2] = 0.275
            desired_mat_list[i] = quaternion_matrix(quaternion_slerp(quat_init, quat_last, i / (step_num - 1)))[:3,
                                  :3].reshape(
                -1)

        desired_posture_list = np.concatenate((desired_pos_list, desired_mat_list), axis=1)
        desired_vel_list = np.array([[0, 0, 0, 0, 0, 0]], dtype=np.float64).repeat(step_num, axis=0)
        desired_acc_list = np.array([[0, 0, 0, 0, 0, 0]], dtype=np.float64).repeat(step_num, axis=0)
        desired_force_list = np.array([[0, 0, 0, 0, 0, 0]], dtype=np.float64).repeat(step_num, axis=0)
        # 初始阻抗参数
        init_M = np.array(np.eye(6), dtype=np.float64)
        init_B = np.array([2000, 2000, 2000, 2000, 2000, 2000], dtype=np.float64)
        init_K = np.array([2000, 2000, 2000, 2000, 2000, 2000], dtype=np.float64)
        min_K = np.array([50, 50, 50, 2000, 2000, 2000], dtype=np.float64)
        max_K = np.array([5000, 5000, 5000, 2000, 2000, 2000], dtype=np.float64)

        env_kwargs = dict(mjc_model_path=mjc_model_path, task=task, qpos_init_list=qpos_init_list,
                          p_bias=p_bias, r_bias=r_bias,
                          step_num=step_num, rl_frequency=rl_frequency, observation_range=observation_range,
                          desired_posture_list=desired_posture_list, desired_vel_list=desired_vel_list,
                          desired_acc_list=desired_acc_list, desired_force_list=desired_force_list,
                          init_M=init_M, init_B=init_B, init_K=init_K, min_K=min_K,
                          max_K=max_K)
        return env_kwargs
    elif task == 'close door':
        # 实验内容
        mjc_model_path = 'robot/jk5_door.xml'
        qpos_init_list = np.array([0, -10, 145, -135, -90, 0]) / 180 * np.pi
        xpos_init_list = np.array([0.38273612, -0.3625067, 0.275])
        p_bias = np.array([0, 0, 0.3885])
        r_bias = quaternion_matrix([0.5, 0.5, 0.5, 0.5])[:3, :3]
        rl_frequency = 250
        observation_range = 1
        step_num = 2000
        # 期望轨迹
        center_pred = np.array([-0.03, -0.4, 0.275])
        handle_pred = np.abs(xpos_init_list[0] - center_pred[0])  # 把手长度预测
        door_pred = np.abs(xpos_init_list[1] - center_pred[1])  # 门板长度预测
        radius_pred = np.linalg.norm(xpos_init_list[:2] - center_pred[:2])  # 半径预测
        angle_pred = np.arctan(door_pred / handle_pred) + np.pi / 2  # 角度预测
        total_angle = np.pi / 2  # 开门角度

        rot_matrix_init = np.eye(4)
        rot_matrix_init[:3, :3] = np.array([[0, 1, 0], [0, 0, -1], [-1, 0, -0]])  # 初始旋转矩阵
        quat_init = quaternion_from_matrix(rot_matrix_init)  # 初始四元数
        rot_matrix_last = np.eye(4)
        rot_matrix_last[:3, :3] = np.array([[0, 0, -1], [0, -1, 0], [-1, 0, 0]])  # 最终旋转矩阵
        quat_last = quaternion_from_matrix(rot_matrix_last)  # 最终四元数
        # 轨迹规划
        desired_pos_list = np.empty((step_num, 3))  # xpos traj
        desired_mat_list = np.empty((step_num, 9))  # xmat traj
        for i in range(step_num):
            desired_pos_list[i][0] = center_pred[0] \
                                     + radius_pred * np.sin(angle_pred - total_angle * i / (step_num - 1))
            desired_pos_list[i][1] = center_pred[1] \
                                     - radius_pred * np.cos(angle_pred - total_angle * i / (step_num - 1))
            desired_pos_list[i][2] = 0.275
            desired_mat_list[i] = quaternion_matrix(quaternion_slerp(quat_init, quat_last, i / (step_num - 1)))[:3,
                                  :3].reshape(-1)

        desired_posture_list = np.concatenate((desired_pos_list, desired_mat_list), axis=1)
        desired_vel_list = np.array([[0, 0, 0, 0, 0, 0]], dtype=np.float64).repeat(step_num, axis=0)
        desired_acc_list = np.array([[0, 0, 0, 0, 0, 0]], dtype=np.float64).repeat(step_num, axis=0)
        desired_force_list = np.array([[0, 0, 0, 0, 0, 0]], dtype=np.float64).repeat(step_num, axis=0)
        # 阻抗参数
        init_M = np.array(np.eye(6), dtype=np.float64)
        init_B = np.array([500, 500, 500, 500, 500, 500], dtype=np.float64)
        init_K = np.array([500, 500, 500, 1000, 1000, 1000], dtype=np.float64)
        min_K = np.array([50, 50, 50, 500, 500, 500], dtype=np.float64)
        max_K = np.array([5000, 5000, 5000, 1000, 1000, 1000], dtype=np.float64)

        env_kwargs = dict(mjc_model_path=mjc_model_path, task=task, qpos_init_list=qpos_init_list,
                          p_bias=p_bias, r_bias=r_bias,
                          step_num=step_num, rl_frequency=rl_frequency, observation_range=observation_range,
                          desired_posture_list=desired_posture_list, desired_vel_list=desired_vel_list,
                          desired_acc_list=desired_acc_list, desired_force_list=desired_force_list,
                          init_M=init_M, init_B=init_B, init_K=init_K, min_K=min_K,
                          max_K=max_K)
        return env_kwargs
    return None


if __name__ == "__main__":
    rbt_kwargs, rl_kwargs = load_env_kwargs('desk')
    model = 1
    if model == 1:
        rbt_kwargs['mjc_model_path'] = '../robot/jk5_table_v2.xml'
        env = Jk5StickStiffnessEnv(**rbt_kwargs)
        s_dict = env.reset()
        contact_force_buffer = [s_dict['contact_force']]
        xpos_buffer = [s_dict['xpos']]
        desired_pos_buffer = [s_dict['desired_pos']]
        for _ in range(rbt_kwargs['step_num']):
            s_dict = env.step(s_dict)
            # env.render(pause_start=True)
            contact_force_buffer.append(s_dict['contact_force'])
            xpos_buffer.append(s_dict['xpos'])
            desired_pos_buffer.append(s_dict['desired_pos'])
        plt.figure(1)
        plt.plot(contact_force_buffer)
        plt.legend(['x', 'y', 'z', 'rx', 'ry', 'rz'])
        plt.grid()
        plt.figure(2)
        plt.plot(xpos_buffer)
        plt.plot(desired_pos_buffer)
        plt.legend(['x', 'y', 'z', 'dx', 'dy', 'dz'])
        plt.grid()
        plt.show()

    else:
        rl_kwargs['mjc_model_path'] = '../robot/jk5_table_v2.xml'
        # rl test
        env = TrainEnv(**rl_kwargs)
        if not check_env(env):
            print('check passed')
        env.reset()
        while True:
            # Random action
            a = env.action_space.sample()
            o, r, d, info = env.step(a)
            env.render(pause_start=True)
            if d:
                break
