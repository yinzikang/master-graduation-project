#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""基于gym.Env的实现，用于sb3环境

继承自v4,分割机器人与控制器

Write typical usage example here

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
3/2/23 11:08 AM   yinzikang      1.0         None
"""
import sys

import numpy

sys.path.append('..')
import PyKDL as kdl
import mujoco as mp
import matplotlib.pyplot as plt

from utils.custom_logx import EpisodeLogger
from utils.custom_viewer import EnvViewer
from module.transformations import quaternion_matrix, quaternion_slerp
from module.controller import *

from gym import spaces, Env
import copy
from spinup.utils.mpi_tools import proc_id
from stable_baselines3.common.env_checker import check_env


# 基础的机器人部分
class Jk5StickRobot:

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
        self.ee_site_name = 'ee'
        self.ee_site_id = mp.mj_name2id(self.mjc_model, mp.mjtObj.mjOBJ_SITE, self.ee_site_name)
        self.dummy_body_name = 'dummy_body'
        self.dummy_body_id = mp.mj_name2id(self.mjc_model, mp.mjtObj.mjOBJ_BODY, self.dummy_body_name)
        # data
        self.data = mp.MjData(self.mjc_model)
        self.status = dict()

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

    @staticmethod
    def mat_to_quat(xmat):
        xmat44 = np.eye(4)
        xmat44[:3, :3] = xmat
        xquat = quaternion_from_matrix(xmat44)
        if xquat[3] < 0:
            xquat = - xquat

        return xquat

    def get_xposture_xvel(self):
        # 位置
        xpos = np.array(self.data.site(self.ee_site_name).xpos)
        # 姿态
        xmat = np.array(self.data.site(self.ee_site_name).xmat).reshape(3, 3)
        xquat = self.mat_to_quat(xmat)
        # 速度
        xvel = np.ndarray(shape=(6,), dtype=np.float64, order='C')
        mp.mj_objectVelocity(self.mjc_model, self.data, mp.mjtObj.mjOBJ_SITE, self.ee_site_id, xvel, False)
        xvel = xvel[[3, 4, 5, 0, 1, 2]]  # 把线速度放在前面
        return xpos, xmat, xquat, xvel

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

    def get_mass_matrix(self):
        # 注意行列数与具体关节数不一定相等，且所需行列不一定相邻
        mass_matrix = np.ndarray(shape=(self.mjc_model.nv, self.mjc_model.nv), dtype=np.float64, order='C')
        mp.mj_fullM(self.mjc_model, mass_matrix, self.data.qM)
        mass_matrix = mass_matrix[:self.joint_num, :self.joint_num]
        return mass_matrix

    def get_jacobian(self):  # 注意行列数与具体关节数不一定相等，且所需行列不一定相邻
        jacp = np.ndarray(shape=(3, self.mjc_model.nv), dtype=np.float64, order='C')
        jacr = np.ndarray(shape=(3, self.mjc_model.nv), dtype=np.float64, order='C')
        mp.mj_jacSite(self.mjc_model, self.data, jacp, jacr, self.ee_site_id)
        J_full = np.array(np.vstack([jacp[:, :self.joint_num], jacr[:, :self.joint_num]]))
        return J_full

    def get_jacobian_dot(self, last_jacobian, cur_jacobian):
        jacobian_d = np.subtract(cur_jacobian, last_jacobian)
        jacobian_d = jacobian_d / self.mjc_model.opt.timestep
        # self.last_jacobian = cur_jacobian
        return jacobian_d

    def get_status(self):
        # 运动学状态
        xpos, xmat, xquat, xvel = self.get_xposture_xvel()
        qpos, qvel = self.get_qpos_qvel()

        J_old = self.status['J']
        J = self.get_jacobian()
        Jd = self.get_jacobian_dot(J_old, J)
        self.status.update(xpos=xpos, xmat=xmat, xquat=xquat, xvel=xvel, qpos=qpos, qvel=qvel, J_old=J_old, J=J, Jd=Jd)

        # 动力学状态：关节空间、笛卡尔空间下的D、C、G
        D_q = self.get_mass_matrix()
        D_x = np.dot(np.linalg.inv(J.T), np.dot(D_q, np.linalg.inv(J)))
        CG_q = np.array(self.data.qfrc_bias[:self.joint_num])
        CG_x = np.dot(np.linalg.inv(J.T), CG_q) - np.dot(np.dot(D_x, Jd), qvel)
        self.status.update(D_q=D_q, D_x=D_x, CG_q=CG_q, CG_x=CG_x)

        # 接触状态：contact_force：笛卡尔空间下，相对于基坐标系，机器人受到的力，按压桌子受到的力朝上，因此contact_force为正
        contact_force = - np.concatenate((xmat[:3, :3] @ self.data.sensor(self.sensor_list[0]).data,
                                          xmat[:3, :3] @ self.data.sensor(self.sensor_list[1]).data))
        touch_force = np.array(self.data.sensor(self.sensor_list[2]).data)
        nft_force = np.concatenate((self.data.sensor(self.sensor_list[3]).data,
                                    self.data.sensor(self.sensor_list[4]).data))
        self.status.update(contact_force=contact_force, touch_force=touch_force, nft_force=nft_force)

        if hasattr(self, 'logger'):
            self.logger.store_buffer(contact_force_series=contact_force, nft_force_series=nft_force)


# 机器人+控制器
class Jk5StickRobotWithController(Jk5StickRobot):
    def __init__(self, mjc_model_path, task, qpos_init_list, p_bias, r_bias,
                 controller_parameter, controller, orientation_error, step_num,
                 desired_xposture_list, desired_xvel_list, desired_xacc_list, desired_force_list):
        super().__init__(mjc_model_path, task, qpos_init_list, p_bias, r_bias)
        # 机器人
        self.initial_state = copy.deepcopy(self.data)
        # 控制器
        self.initial_controller_parameter = controller_parameter
        self.controller_parameter = copy.deepcopy(self.initial_state)
        self.controller = controller(orientation_error)

        self.desired_xposture_list = desired_xposture_list.copy()
        self.desired_xvel_list = desired_xvel_list.copy()
        self.desired_xacc_list = desired_xacc_list.copy()
        self.desired_force_list = desired_force_list.copy()

        self.step_num = step_num
        self.current_step = 0

    def get_status(self):
        super().get_status()
        self.status.update(controller_parameter=self.controller_parameter.copy(),
                           desired_xpos=self.desired_xposture_list[self.current_step][:3],
                           desired_xmat=self.desired_xposture_list[self.current_step][3:].reshape(3, 3),
                           desired_xquat=self.mat_to_quat(
                               self.desired_xposture_list[self.current_step][3:].reshape(3, 3)),
                           desired_xvel=self.desired_xvel_list[self.current_step],
                           desired_xacc=self.desired_xacc_list[self.current_step],
                           tau=np.array(self.data.ctrl[:]))

        if hasattr(self, 'logger'):
            self.logger.store_buffer(K_series=self.status["controller_parameter"].copy(),
                                     tau_series=self.status["tau"].copy())

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

        # impedance control reset
        self.controller_parameter = copy.deepcopy(self.initial_controller_parameter)

        self.status.update(J=self.get_jacobian())
        self.status.update(timestep=self.mjc_model.opt.timestep)
        self.get_status()
        # algorithm reset ##################################################################
        # 步数更新
        self.current_step = 0

    def step(self):
        """
        控制器运行一次
        """
        # self.data.xfrc_applied[mp.mj_name2id(self.mjc_model, mp.mjtObj.mjOBJ_BODY, 'dummy_body')][4] = 50
        tau = self.controller.step(self.status)
        # 执行
        self.data.ctrl[:] = tau
        mp.mj_step2(self.mjc_model, self.data)
        mp.mj_step1(self.mjc_model, self.data)

        self.get_status()
        self.current_step += 1

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


# 机器人变阻抗控制
class TrainEnv(Jk5StickRobotWithController, Env):

    def __init__(self, mjc_model_path, task, qpos_init_list, p_bias, r_bias,
                 controller_parameter, controller, orientation_error, step_num,
                 desired_xposture_list, desired_xvel_list, desired_xacc_list, desired_force_list,
                 min_K, max_K, rl_frequency, observation_range):
        super().__init__(mjc_model_path, task, qpos_init_list, p_bias, r_bias,
                         controller_parameter, controller, orientation_error, step_num,
                         desired_xposture_list, desired_xvel_list, desired_xacc_list, desired_force_list)

        self.min_K = min_K.copy()
        self.max_K = max_K.copy()
        M = np.diagonal(controller_parameter['M'])[0]
        B = controller_parameter['B'][0]
        K = controller_parameter['K'][0]
        self.wn = np.sqrt(K / M)
        self.damping_ratio = B / 2 / np.sqrt(K * M)
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
                                            high=np.inf * np.ones(self.observation_num),
                                            dtype=np.float32)  # 连续观测空间
        self.action_space = spaces.Box(low=-1 * np.ones(self.action_num),
                                       high=1 * np.ones(self.action_num),
                                       dtype=np.float32)  # 连续动作空间
        self.action_limit = 50

        self.current_episode = -1

    def get_observation(self):
        xpos = self.status['xpos']
        xquat = self.status['xquat']
        xpos_vel = self.status['xvel'][:3]

        status = np.concatenate((xpos, xquat, xpos_vel), dtype=np.float32)
        self.observation_buffer.append(status)  # status与observation关系
        # 够长则取最近的observation_range个，不够对最远那个进行复制，observation_range×(3+4+2)
        if len(self.observation_buffer) >= self.observation_range:
            observation = np.array(self.observation_buffer[-self.observation_range:]).flatten()
        else:
            observation = np.array(self.observation_buffer +
                                   self.observation_buffer[0] * (
                                           self.current_step - self.observation_range)).flatten()
        return observation

    def get_reward(self):
        xpos_error = self.status['desired_xpos'] - self.status['xpos']
        contact_force = self.status['contact_force']
        tau = self.status['tau']
        done = self.status['done']
        failure = self.status['failure']

        ## 运动状态的奖励
        movement_reward = np.sum(xpos_error[[0, 1]] ** 2)
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

        # reward = 0 * movement_reward + 0.05 * fext_reward + 0 * tau_reward + \
        #           1 * error_k_penalty + 1 * early_stop_penalty + \
        #           0.5 * zero_force_penalty + 0 * massive_force_penalty
        reward = 0 * movement_reward + 0.05 * fext_reward + 0 * tau_reward + \
                 0 * error_k_penalty + 0 * early_stop_penalty + \
                 0 * zero_force_penalty + 0 * massive_force_penalty
        return reward

    def reset(self):
        super().reset()
        self.current_episode += 1
        self.observation_buffer = []

        return self.get_observation()

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
        sub_step = 0
        for sub_step in range(self.sub_step_num):
            # action，即刚度变化量，进行插值
            self.controller_parameter['K'] += self.action_limit * action / self.sub_step_num  # 只有z方向
            M = self.controller_parameter['K'] / (self.wn * self.wn)
            self.controller_parameter['B'] = 2 * self.damping_ratio * numpy.sqrt(M * self.controller_parameter['K'])
            self.controller_parameter['M'] = np.diag(M)
            super().step()
            # 可视化
            if hasattr(self, 'viewer'):
                self.viewer.render()

            # 刚度越界或者任务结束，视为done
            if self.current_step + 1 == self.step_num:
                success = True
                other_info['is_success'], other_info["TimeLimit.truncated"], other_info[
                    'terminal info'] = True, True, 'success'
            else:
                success = False
            if any(np.greater(self.controller_parameter['K'], self.max_K)) or \
                    any(np.greater(self.min_K, self.controller_parameter['K'])):
                failure = True
                other_info['is_success'], other_info["TimeLimit.truncated"], other_info[
                    'terminal info'] = False, False, 'error K'
            else:
                failure = False
            done = success or failure
            self.status.update(done=done, success=success, failure=failure)
            # 获得奖励
            reward += self.get_reward()

            if done:
                break

        reward = reward / (sub_step + 1)
        if hasattr(self, 'logger'):
            self.logger.store_buffer(action_series=action, reward_series=reward)

        return self.get_observation(), reward, done, other_info

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
        mjc_model_path = 'robot/jk5_table_v2.xml'
        qpos_init_list = np.array([0, -30, 120, 0, -90, 0]) / 180 * np.pi
        p_bias = np.zeros(3)
        r_bias = np.eye(3)
        rl_frequency = 20  # 500可以整除，越大越多
        observation_range = 1
        step_num = 2000
        # 期望轨迹
        # desired_xpos_list = np.concatenate((np.linspace(-0.45, -0.75, step_num).reshape(step_num, 1),
        #                                     -0.1135 * np.ones((step_num, 1), dtype=float),
        #                                     0.05 * np.ones((step_num, 1), dtype=float)), axis=1)
        # desired_mat_list = np.array([[0, -1, 0, -1, 0, 0, 0, 0, -1]], dtype=np.float64).repeat(step_num, axis=0)
        # desired_xposture_list = np.concatenate((desired_xpos_list, desired_mat_list), axis=1)
        # desired_xvel_list = np.array([[-0.3 / step_num, 0, 0, 0, 0, 0]], dtype=np.float64).repeat(step_num, axis=0)
        # desired_xacc_list = np.array([[0, 0, 0, 0, 0, 0]], dtype=np.float64).repeat(step_num, axis=0)
        # desired_force_list = np.array([[0, 0, 30, 0, 0, 0]], dtype=np.float64).repeat(step_num, axis=0)
        desired_xpos_list = np.concatenate((np.linspace(-0.5, -0.5, step_num).reshape(step_num, 1),
                                            -0.1135 * np.ones((step_num, 1), dtype=float),
                                            0.05 * np.ones((step_num, 1), dtype=float)), axis=1)
        desired_mat_list = np.array([[0, -1, 0, -1, 0, 0, 0, 0, -1]], dtype=np.float64).repeat(step_num, axis=0)
        desired_xposture_list = np.concatenate((desired_xpos_list, desired_mat_list), axis=1)
        desired_xvel_list = np.array([[-0. / step_num, 0, 0, 0, 0, 0]], dtype=np.float64).repeat(step_num, axis=0)
        desired_xacc_list = np.array([[0, 0, 0, 0, 0, 0]], dtype=np.float64).repeat(step_num, axis=0)
        desired_force_list = np.array([[0, 0, 30, 0, 0, 0]], dtype=np.float64).repeat(step_num, axis=0)
        # 阻抗参数
        wn = 20
        damping_ratio = np.sqrt(2)
        K = np.array([1000, 1000, 1000, 1000, 1000, 1000], dtype=np.float64)
        M = K / (wn * wn)
        M_matrix = np.diag(M)
        B = 2 * damping_ratio * np.sqrt(M * K)
        controller_parameter = {'M': M_matrix, 'B': B, 'K': K}
        controller = ImpedanceController
        orientation_error = orientation_error_axis_angle
        min_K = np.array([100, 100, 100, 100, 100, 100], dtype=np.float64)
        max_K = np.array([3000, 3000, 3000, 3000, 3000, 3000], dtype=np.float64)
        rbt_kwargs = dict(mjc_model_path=mjc_model_path, task=task, qpos_init_list=qpos_init_list,
                          p_bias=p_bias, r_bias=r_bias)
        # 用于Jk5StickStiffnessEnv的超参数
        rbt_controller_kwargs = copy.deepcopy(rbt_kwargs)
        rbt_controller_kwargs.update(controller_parameter=controller_parameter, controller=controller,
                                     orientation_error=orientation_error, step_num=step_num,
                                     desired_xposture_list=desired_xposture_list, desired_xvel_list=desired_xvel_list,
                                     desired_xacc_list=desired_xacc_list, desired_force_list=desired_force_list)

        # 用于TrainEnv的超参数
        rl_env_kwargs = copy.deepcopy(rbt_controller_kwargs)
        rl_env_kwargs.update(min_K=min_K, max_K=max_K,
                             rl_frequency=rl_frequency, observation_range=observation_range)

        return rbt_kwargs, rbt_controller_kwargs, rl_env_kwargs
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
        desired_xpos_list = np.empty((step_num, 3))  # xpos traj
        desired_mat_list = np.empty((step_num, 9))  # xmat traj
        for i in range(step_num):
            desired_xpos_list[i][0] = center_pred[0] + \
                                      radius_pred * np.sin(angle_pred + total_angle * i / (step_num - 1))
            desired_xpos_list[i][1] = center_pred[1] \
                                      - radius_pred * np.cos(angle_pred + total_angle * i / (step_num - 1))
            desired_xpos_list[i][2] = 0.275
            desired_mat_list[i] = quaternion_matrix(quaternion_slerp(quat_init, quat_last, i / (step_num - 1)))[:3,
                                  :3].reshape(
                -1)

        desired_xposture_list = np.concatenate((desired_xpos_list, desired_mat_list), axis=1)
        desired_xvel_list = np.array([[0, 0, 0, 0, 0, 0]], dtype=np.float64).repeat(step_num, axis=0)
        desired_xacc_list = np.array([[0, 0, 0, 0, 0, 0]], dtype=np.float64).repeat(step_num, axis=0)
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
                          desired_xposture_list=desired_xposture_list, desired_xvel_list=desired_xvel_list,
                          desired_xacc_list=desired_xacc_list, desired_force_list=desired_force_list,
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
        desired_xpos_list = np.empty((step_num, 3))  # xpos traj
        desired_mat_list = np.empty((step_num, 9))  # xmat traj
        for i in range(step_num):
            desired_xpos_list[i][0] = center_pred[0] \
                                      + radius_pred * np.sin(angle_pred - total_angle * i / (step_num - 1))
            desired_xpos_list[i][1] = center_pred[1] \
                                      - radius_pred * np.cos(angle_pred - total_angle * i / (step_num - 1))
            desired_xpos_list[i][2] = 0.275
            desired_mat_list[i] = quaternion_matrix(quaternion_slerp(quat_init, quat_last, i / (step_num - 1)))[:3,
                                  :3].reshape(-1)

        desired_xposture_list = np.concatenate((desired_xpos_list, desired_mat_list), axis=1)
        desired_xvel_list = np.array([[0, 0, 0, 0, 0, 0]], dtype=np.float64).repeat(step_num, axis=0)
        desired_xacc_list = np.array([[0, 0, 0, 0, 0, 0]], dtype=np.float64).repeat(step_num, axis=0)
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
                          desired_xposture_list=desired_xposture_list, desired_xvel_list=desired_xvel_list,
                          desired_xacc_list=desired_xacc_list, desired_force_list=desired_force_list,
                          init_M=init_M, init_B=init_B, init_K=init_K, min_K=min_K,
                          max_K=max_K)
        return env_kwargs
    return None


if __name__ == "__main__":
    rbt_kwargs, rbt_controller_kwargs, rl_kwargs = load_env_kwargs('desk')
    mode = 1  # 0为机器人，1为机器人+控制器，2为强化学习用于变阻抗控制
    controller = 1  # 0为阻抗控制，1为导纳控制，2为计算力矩控制
    orientation_error = 1  # 0为轴角旋转误差，1为四元数旋转误差
    if mode == 0:
        rbt_kwargs['mjc_model_path'] = '../robot/jk5_table_v2.xml'
        rbt = Jk5StickRobot(**rbt_kwargs)
    elif mode == 1:
        rbt_controller_kwargs['mjc_model_path'] = '../robot/jk5_table_v2.xml'
        # rbt_controller_kwargs['qpos_init_list'] = np.array([0, -30, 120, 0, -60, 0]) / 180 * np.pi
        if controller == 1:
            rbt_controller_kwargs['controller'] = AdmittanceController
        elif controller == 2:
            wn = 20
            damping_ratio = np.sqrt(2)
            kp = wn * wn * np.ones(6, dtype=np.float64)
            kd = 2 * damping_ratio * numpy.sqrt(kp)
            rbt_controller_kwargs['controller_parameter'] = {'kp': kp, 'kd': kd}
            rbt_controller_kwargs['controller'] = ComputedTorqueController
        if orientation_error == 1:
            rbt_controller_kwargs['orientation_error'] = orientation_error_quaternion
        env = Jk5StickRobotWithController(**rbt_controller_kwargs)
        env.reset()
        xpos_buffer = [env.status['xpos']]
        xmat_buffer = [env.status['xmat']]
        xquat_buffer = [env.status['xquat']]
        desired_xpos_buffer = [env.status['desired_xpos']]
        desired_xmat_buffer = [env.status['desired_xmat']]
        desired_xquat_buffer = [env.status['desired_xquat']]
        contact_force_buffer = [env.status['contact_force']]
        tau_buffer = [env.status['tau']]
        for _ in range(rbt_controller_kwargs['step_num']):
            env.step()
            # env.render(pause_start=True)

            xpos_buffer.append(env.status['xpos'])
            xmat_buffer.append(env.status['xmat'])
            xquat_buffer.append(env.status['xquat'])
            desired_xpos_buffer.append(env.status['desired_xpos'])
            desired_xmat_buffer.append(env.status['desired_xmat'])
            desired_xquat_buffer.append(env.status['desired_xquat'])
            contact_force_buffer.append(env.status['contact_force'])
            tau_buffer.append(env.status['tau'])

        fig_title = rbt_controller_kwargs['controller'].__name__ + ' ' + \
                    rbt_controller_kwargs['orientation_error'].__name__ + ' '

        i = 0

        # i += 1
        # plt.figure(i)
        # plt.plot(xpos_buffer)
        # plt.plot(desired_xpos_buffer)
        # plt.legend(['x', 'y', 'z', 'dx', 'dy', 'dz'])
        # plt.title(fig_title + 'xpos')
        # plt.grid()

        # i += 1
        # plt.figure(i)
        # plt.plot((np.array(xpos_buffer) - np.array(desired_xpos_buffer))[1:, :] /
        #          np.array(contact_force_buffer)[1:, :3])
        # plt.legend(['x', 'y', 'z'])
        # plt.title(fig_title + '1/stiffness')
        # plt.grid()

        i += 1
        plt.figure(i)
        plt.plot(xquat_buffer)
        plt.plot(desired_xquat_buffer)
        plt.legend(['x', 'y', 'z', 'w', 'dx', 'dy', 'dz', 'dw'])
        plt.title(fig_title + 'xquat')
        plt.grid()

        i += 1
        plt.figure(i)
        orientation_error_buffer = []
        for j in range(len(xquat_buffer)):
            orientation_error_buffer.append(
                Jk5StickRobotWithController.mat_to_quat(np.linalg.inv(desired_xmat_buffer[j]) @ xmat_buffer[j]))
        plt.plot(orientation_error_buffer)
        plt.legend(['x', 'y', 'z', 'w'])
        plt.title(fig_title + 'orientation_error')
        plt.grid()

        i += 1
        plt.figure(i)
        plt.plot(contact_force_buffer)
        plt.legend(['x', 'y', 'z', 'rx', 'ry', 'rz'])
        plt.title(fig_title + 'force')
        plt.grid()

        # i += 1
        # plt.figure(i)
        # plt.plot(tau_buffer)
        # plt.legend(['x', 'y', 'z', 'rx', 'ry', 'rz'])
        # plt.title(fig_title+'tau')
        # plt.grid()

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
