#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""基于gym.Env的实现，用于sb3环境

继承自v6,添加姿态动作
log更加合理

Write typical usage example here

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
3/2/23 11:08 AM   yinzikang      6.0         None
"""

import numpy as np
import mujoco as mp
import PyKDL as kdl
from utils.custom_logx import EpisodeLogger
from utils.custom_viewer import EnvViewer
from module.controller import mat33_to_quat, to_kdl_qpos, to_numpy_qpos, to_kdl_frame, to_numpy_frame
from gym import spaces, Env
from module.transformations import quaternion_about_axis, rotation_matrix, quaternion_multiply
import copy
from spinup.utils.mpi_tools import proc_id


# 基础的机器人部分
class Jk5StickRobot:

    def __init__(self, mjc_model_path, task, qpos_init_list, p_bias, r_bias):
        # robot part #########################################################################
        self.joint_num = 6
        self.qpos_init_list = qpos_init_list
        self.task = task
        # 两个模型
        self.mjc_model = mp.MjModel.from_xml_path(filename=mjc_model_path, assets=None)
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
        self.status_list = ['xpos', 'xmat', 'xquat', 'xvel', 'qpos', 'qvel', 'J_old', 'J', 'Jd',
                            'contact_force', 'touch_force', 'nft_force']

    def create_kdl_model(self):
        """
        kdl机器人模型
        用于搭配运动学内容，包括轨迹规划，前向运动学，逆向运动学
        组成包括机器人本体，以及连杆，即力传感器位置，并没有设计到工具
        :return:
        """
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
        elif self.task == 'open door':  # 门任务ee长0.06
            rbt.addSegment(kdl.Segment("end-effector", kdl.Joint("ee_joint", kdl.Joint.Fixed),
                                       kdl.Frame(kdl.Rotation.Identity(), kdl.Vector(0, 0, 0.05))))
        elif self.task == 'close door':  # 门任务ee长0.06
            rbt.addSegment(kdl.Segment("end-effector", kdl.Joint("ee_joint", kdl.Joint.Fixed),
                                       kdl.Frame(kdl.Rotation.Identity(), kdl.Vector(0, 0, 0.063))))
        elif self.task == 'cabinet surface with plan':  # 储物柜桌面任务ee长0.169
            rbt.addSegment(kdl.Segment("end-effector", kdl.Joint("ee_joint", kdl.Joint.Fixed),
                                       kdl.Frame(kdl.Rotation.Identity(), kdl.Vector(0, 0, 0.169))))
        elif self.task == 'cabinet drawer open' or self.task == 'cabinet drawer close':  # 执行器中心
            rbt.addSegment(kdl.Segment("end-effector", kdl.Joint("ee_joint", kdl.Joint.Fixed),
                                       kdl.Frame(kdl.Rotation.Identity(), kdl.Vector(0, 0, 0.169))))
        elif self.task == 'cabinet door open' or self.task == 'cabinet door close':  # 储物柜门任务ee长0.06
            rbt.addSegment(kdl.Segment("end-effector", kdl.Joint("ee_joint", kdl.Joint.Fixed),
                                       kdl.Frame(kdl.Rotation.Identity(), kdl.Vector(0, 0, 0.169))))

        return rbt

    def get_xposture_xvel(self):
        # 位置
        xpos = np.array(self.data.site(self.ee_site_name).xpos)
        # 姿态
        xmat = np.array(self.data.site(self.ee_site_name).xmat).reshape(3, 3)
        xquat = mat33_to_quat(xmat)
        # 速度，qvel中，线速度是相对于世界坐标系的，而角速度是相对于自身坐标系的m，以下函数可以设置角速度的参考系，当前为世界坐标系
        xvel = np.ndarray(shape=(6,), dtype=np.float64, order='C')
        mp.mj_objectVelocity(self.mjc_model, self.data, mp.mjtObj.mjOBJ_SITE, self.ee_site_id, xvel, flg_local=False)
        xvel = xvel[[3, 4, 5, 0, 1, 2]]  # 把线速度放在前面

        return xpos, xmat, xquat, xvel

    def get_qpos_qvel(self):
        qpos = np.array([self.data.joint(joint_name).qpos for joint_name in self.joint_list]).reshape(-1)
        qvel = np.array([self.data.joint(joint_name).qvel for joint_name in self.joint_list]).reshape(-1)
        return qpos, qvel

    def forward_kinematics(self, qpos_list):
        kdl_qpos_list = to_kdl_qpos(self.joint_num, qpos_list)
        kdl_frame = kdl.Frame()
        fksolver = kdl.ChainFkSolverPos_recursive(self.kdl_model)
        fksolver.JntToCart(kdl_qpos_list, kdl_frame)
        pos, mat = to_numpy_frame(kdl_frame)
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
        kdl_qpos_init_list = to_kdl_qpos(self.joint_num, qpos_init_list)
        kdl_frame = to_kdl_frame(xpos, xmat)
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
        joint_pos_list = to_numpy_qpos(self.joint_num, kdl_joint_pos)

        return joint_pos_list

    def get_mass_matrix(self):
        # 注意行列数与具体关节数不一定相等，且所需行列不一定相邻
        mass_matrix = np.ndarray(shape=(self.mjc_model.nv, self.mjc_model.nv), dtype=np.float64, order='C')
        mp.mj_fullM(self.mjc_model, mass_matrix, self.data.qM)
        mass_matrix = mass_matrix[:self.joint_num, :self.joint_num]
        return mass_matrix

    def get_jacobian(self):
        """
        相对于ee_site的雅克比，用于力矩的空间换算
        注意行列数与具体关节数不一定相等，且所需行列不一定相邻
        """
        jacp = np.ndarray(shape=(3, self.mjc_model.nv), dtype=np.float64, order='C')
        jacr = np.ndarray(shape=(3, self.mjc_model.nv), dtype=np.float64, order='C')
        mp.mj_jacSite(self.mjc_model, self.data, jacp, jacr, self.ee_site_id)
        J_full = np.array(np.vstack([jacp[:, :self.joint_num], jacr[:, :self.joint_num]]))
        return J_full

    def get_jacobian_dot(self, last_jacobian, cur_jacobian):
        jacobian_d = np.subtract(cur_jacobian, last_jacobian)
        jacobian_d = jacobian_d / self.mjc_model.opt.timestep
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


# 机器人+控制器
class Jk5StickRobotWithController(Jk5StickRobot):
    def __init__(self, mjc_model_path, task, qpos_init_list, p_bias, r_bias,
                 controller_parameter, controller, step_num,
                 desired_xposture_list, desired_xvel_list, desired_xacc_list, desired_force_list):
        super().__init__(mjc_model_path, task, qpos_init_list, p_bias, r_bias)
        # 机器人
        self.initial_state = copy.deepcopy(self.data)
        # 控制器
        self.initial_controller_parameter = controller_parameter
        self.controller_parameter = copy.deepcopy(self.initial_state)
        self.controller = controller()

        self.init_desired_xposture_list = desired_xposture_list
        self.init_desired_xvel_list = desired_xvel_list
        self.init_desired_xacc_list = desired_xacc_list
        self.init_desired_force_list = desired_force_list
        self.desired_xposture_list = copy.deepcopy(self.init_desired_xposture_list)
        self.desired_xvel_list = copy.deepcopy(self.init_desired_xvel_list)
        self.desired_xacc_list = copy.deepcopy(self.init_desired_xacc_list)
        self.desired_force_list = copy.deepcopy(self.init_desired_force_list)

        self.status_list.extend(['controller_parameter',
                                 'desired_xpos', 'desired_xmat', 'desired_xquat', 'desired_xvel', 'desired_xacc',
                                 'desired_force', 'tau'])

        self.step_num = step_num
        self.current_step = 0

    def get_status(self):
        super().get_status()
        self.status.update(controller_parameter=self.controller_parameter,
                           desired_xpos=self.desired_xposture_list[self.current_step][:3],
                           desired_xmat=self.desired_xposture_list[self.current_step][3:12].reshape(3, 3),
                           desired_xquat=self.desired_xposture_list[self.current_step][12:16],
                           desired_xvel=self.desired_xvel_list[self.current_step],
                           desired_xacc=self.desired_xacc_list[self.current_step],
                           desired_force=self.desired_force_list[self.current_step],
                           tau=np.array(self.data.ctrl[:]),
                           timestep=self.mjc_model.opt.timestep,
                           current_step = self.current_step)

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

        if self.task == "cabinet drawer open with plan":
            pos1 = np.array(self.data.body("dummy_body").xpos)
            pos2 = np.array(self.data.body("drawer handle").xpos)
            mat1 = np.array(self.data.body("dummy_body").xmat).reshape(3, 3)
            pos_delta2 = np.matmul(np.linalg.inv(mat1), (pos2 - pos1).reshape(3, 1)).reshape(-1)
            self.mjc_model.equality('robot2drawer').data = np.hstack((pos_delta2, np.zeros(4)))
            self.mjc_model.equality('robot2drawer').active = True

        if self.task == "cabinet drawer close":
            self.data.joint('drawer joint').qpos = 0.3

        mp.mj_forward(self.mjc_model, self.data)

        # impedance control reset #####################################################################
        self.controller_parameter = copy.deepcopy(self.initial_controller_parameter)
        self.desired_xposture_list = copy.deepcopy(self.init_desired_xposture_list)
        self.desired_xvel_list = copy.deepcopy(self.init_desired_xvel_list)
        self.desired_xacc_list = copy.deepcopy(self.init_desired_xacc_list)
        self.desired_force_list = copy.deepcopy(self.init_desired_force_list)
        self.status.update(J=self.get_jacobian())
        self.current_step = 0
        self.get_status()

    def step(self):
        """
        控制器运行一次
        """
        # self.data.xfrc_applied[mp.mj_name2id(self.mjc_model, mp.mjtObj.mjOBJ_BODY, 'dummy_body')][4] = -20
        tau = self.controller.step(self.status)
        # 执行
        self.data.ctrl[:] = tau
        mp.mj_step2(self.mjc_model, self.data)
        mp.mj_step1(self.mjc_model, self.data)

        self.current_step += 1
        self.get_status()

    def logger_init(self, output_dir=None):
        if proc_id() == 0:
            self.logger = EpisodeLogger(output_dir)

    def viewer_init(self, pause_start=False, view_force=True):
        if proc_id() == 0:
            self.viewer = EnvViewer(self, pause_start, view_force)

    def render(self, mode="human", pause_start=False, view_force=True):
        if not hasattr(self, 'viewer'):
            self.viewer_init(pause_start, view_force)
        self.viewer.render()


# 机器人变阻抗控制
class TrainEnvBase(Jk5StickRobotWithController, Env):
    def __init__(self, mjc_model_path, task, qpos_init_list, p_bias, r_bias,
                 controller_parameter, controller, step_num,
                 desired_xposture_list, desired_xvel_list, desired_xacc_list, desired_force_list,
                 min_K, max_K, rl_frequency, observation_range):
        super().__init__(mjc_model_path, task, qpos_init_list, p_bias, r_bias,
                         controller_parameter, controller, step_num,
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
        self.observation_num = 3 + 3  # 观测数：3个位置+3个速度

        self.current_episode = -1

    def get_observation(self):
        xpos = self.status['xpos']
        xquat = self.status['xquat']
        xpos_vel = self.status['xvel'][:3]

        status = np.concatenate((xpos, xpos_vel), dtype=np.float32)
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
        # fext_reward = - np.sum(
        #     np.linalg.norm(contact_force[2] - self.desired_force_list[self.current_step, 2], ord=1))
        fext_reward = - abs(contact_force[2] - self.desired_force_list[self.current_step, 2])
        fext_reward = fext_reward + 10 if fext_reward > -2.5 else fext_reward  # 要是力距离期望力较近则进行额外奖励
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
                 0 * zero_force_penalty + 0 * massive_force_penalty + 1.
        return reward

    def reset(self):
        super().reset()
        self.current_episode += 1
        self.observation_buffer = []

        return self.get_observation()

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


class TrainEnvVariableStiffness(TrainEnvBase):
    def __init__(self, mjc_model_path, task, qpos_init_list, p_bias, r_bias,
                 controller_parameter, controller, step_num,
                 desired_xposture_list, desired_xvel_list, desired_xacc_list, desired_force_list,
                 min_K, max_K, rl_frequency, observation_range):
        super().__init__(mjc_model_path, task, qpos_init_list, p_bias, r_bias,
                         controller_parameter, controller, step_num,
                         desired_xposture_list, desired_xvel_list, desired_xacc_list, desired_force_list,
                         min_K, max_K, rl_frequency, observation_range)
        self.action_num = 6  # 动作数：刚度变化量
        self.observation_space = spaces.Box(low=-np.inf * np.ones(self.observation_num),
                                            high=np.inf * np.ones(self.observation_num),
                                            dtype=np.float32)  # 连续观测空间
        self.action_space = spaces.Box(low=-1 * np.ones(self.action_num),
                                       high=1 * np.ones(self.action_num),
                                       dtype=np.float32)  # 连续动作空间
        self.action_limit = np.array([100, 100, 100, 10, 10, 10], dtype=np.float32)

    def step(self, action):
        """
        运行动作，获得下一次观测
        :param action:
        :return:
        """
        # 其余状态的初始化
        reward = 0
        done, success, error_K, error_force = False, False, False, False
        other_info = dict()
        # 对多次执行机器人控制
        sub_step = 0
        for sub_step in range(self.sub_step_num):
            # action，即刚度变化量，进行插值
            self.controller_parameter['K'] += self.action_limit * action / self.sub_step_num  # 只有z方向
            M = self.controller_parameter['K'] / (self.wn * self.wn)
            self.controller_parameter['B'] = 2 * self.damping_ratio * np.sqrt(M * self.controller_parameter['K'])
            self.controller_parameter['M'] = np.diag(M)
            # print(self.controller_parameter['K'])
            super().step()
            # 可视化
            if hasattr(self, 'viewer'):
                self.viewer.render()
            if hasattr(self, 'logger'):
                self.logger.store_buffer(xpos=self.status["xpos"].copy(), xmat=self.status["xmat"].copy(),
                                         xquat=self.status["xquat"].copy(), xvel=self.status["xvel"].copy(),
                                         qpos=self.status["qpos"].copy(), qvel=self.status["qvel"].copy(),
                                         contact_force=self.status["contact_force"].copy(),
                                         nft_force=self.status["nft_force"].copy(),
                                         K=self.controller_parameter['K'].copy(),
                                         desired_xpos=self.status["desired_xpos"].copy(),
                                         desired_xmat=self.status["desired_xmat"].copy(),
                                         desired_xquat=self.status["desired_xquat"].copy(),
                                         desired_xvel=self.status["desired_xvel"].copy(),
                                         desired_xacc=self.status["desired_xacc"].copy(),
                                         desired_force=self.status["desired_force"].copy(),
                                         tau=self.status["tau"].copy())

            success, error_K, error_force = False, False, False
            # 时间约束：到达最大时间，视为done
            if self.current_step + 1 == self.step_num:
                success = True
                other_info['is_success'], other_info["TimeLimit.truncated"], other_info[
                    'terminal info'] = True, True, 'success'
            # 刚度约束：到达最大时间，视为done
            if any(np.greater(self.controller_parameter['K'], self.max_K)) or \
                    any(np.greater(self.min_K, self.controller_parameter['K'])):
                error_K = True
                other_info['is_success'], other_info["TimeLimit.truncated"], other_info[
                    'terminal info'] = False, False, 'error K'
                print(self.controller_parameter['K'])
            # 接触力约束
            if any(np.greater(self.status['contact_force'], np.array([50, 50, 50, 50, 50, 50]))):
                error_force = True
                other_info['is_success'], other_info["TimeLimit.truncated"], other_info[
                    'terminal info'] = False, False, 'error force'
                print(self.status['contact_force'])
            failure = error_K or error_force
            done = success or failure

            # 获得奖励
            reward += self.get_reward(done, success, failure)

            if done:
                break

        reward = reward / (sub_step + 1)
        if hasattr(self, 'logger'):
            self.logger.store_buffer(action=action, reward=reward)

        return self.get_observation(), reward, done, other_info


class TrainEnvVariableStiffnessAndPosture(TrainEnvBase):
    def __init__(self, mjc_model_path, task, qpos_init_list, p_bias, r_bias,
                 controller_parameter, controller, step_num,
                 desired_xposture_list, desired_xvel_list, desired_xacc_list, desired_force_list,
                 min_K, max_K, rl_frequency, observation_range):
        super().__init__(mjc_model_path, task, qpos_init_list, p_bias, r_bias,
                         controller_parameter, controller, step_num,
                         desired_xposture_list, desired_xvel_list, desired_xacc_list, desired_force_list,
                         min_K, max_K, rl_frequency, observation_range)
        self.action_num = 6 + 3 + 4  # 动作数：刚度变化量+位姿变化量
        self.observation_space = spaces.Box(low=-np.inf * np.ones(self.observation_num),
                                            high=np.inf * np.ones(self.observation_num),
                                            dtype=np.float32)  # 连续观测空间
        self.action_space = spaces.Box(low=-1 * np.ones(self.action_num),
                                       high=1 * np.ones(self.action_num),
                                       dtype=np.float32)  # 连续动作空间
        self.action_limit = np.array([100, 100, 100, 10, 10, 10,
                                      0.01, 0.01, 0.01,  # 位置变化限制
                                      1, 1, 1,  # 旋转轴变化限制，无意义，反正会标准化
                                      0.001], dtype=np.float32)  # 姿态的角度变化限制

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
        action = self.action_limit * action
        for sub_step in range(self.sub_step_num):
            # action，即刚度变化量，进行插值
            self.controller_parameter['K'] += action[:6] / self.sub_step_num  # 只有z方向
            M = self.controller_parameter['K'] / (self.wn * self.wn)
            self.controller_parameter['B'] = 2 * self.damping_ratio * np.sqrt(M * self.controller_parameter['K'])
            self.controller_parameter['M'] = np.diag(M)
            # print(self.controller_parameter['K'])

            pos, direction, angle = action[6:9], action[9:12] / np.linalg.norm(action[9:12]), action[12]
            mat = rotation_matrix(angle, direction)[:3, :3]
            quat = quaternion_about_axis(angle, direction)
            self.status['desired_xpos'] += pos
            self.status['desired_xmat'] = mat @ self.status['desired_xmat']
            self.status['desired_xquat'] = quaternion_multiply(quat, self.status['desired_xquat'])

            # 可视化
            if hasattr(self, 'viewer'):
                self.viewer.render()
            if hasattr(self, 'logger'):
                self.logger.store_buffer(xpos=self.status["xpos"].copy(), xmat=self.status["xmat"].copy(),
                                         xquat=self.status["xquat"].copy(), xvel=self.status["xvel"].copy(),
                                         qpos=self.status["qpos"].copy(), qvel=self.status["qvel"].copy(),
                                         contact_force=self.status["contact_force"].copy(),
                                         nft_force=self.status["nft_force"].copy(),
                                         K=self.controller_parameter['K'].copy(),
                                         desired_xpos=self.status["desired_xpos"].copy(),
                                         desired_xmat=self.status["desired_xmat"].copy(),
                                         desired_xquat=self.status["desired_xquat"].copy(),
                                         desired_xvel=self.status["desired_xvel"].copy(),
                                         desired_xacc=self.status["desired_xacc"].copy(),
                                         desired_force=self.status["desired_force"].copy(),
                                         tau=self.status["tau"].copy())

            super().step()
            success, error_K, error_force = False, False, False
            # 时间约束：到达最大时间，视为done
            if self.current_step + 1 == self.step_num:
                success = True
                other_info['is_success'], other_info["TimeLimit.truncated"], other_info[
                    'terminal info'] = True, True, 'success'
            # 刚度约束：到达最大时间，视为done
            if any(np.greater(self.controller_parameter['K'], self.max_K)) or \
                    any(np.greater(self.min_K, self.controller_parameter['K'])):
                error_K = True
                other_info['is_success'], other_info["TimeLimit.truncated"], other_info[
                    'terminal info'] = False, False, 'error K'
                print(self.controller_parameter['K'])
            # 接触力约束
            if any(np.greater(self.status['contact_force'], np.array([50, 50, 50, 50, 50, 50]))):
                error_force = True
                other_info['is_success'], other_info["TimeLimit.truncated"], other_info[
                    'terminal info'] = False, False, 'error force'
                print(self.status['contact_force'])
            failure = error_K or error_force
            done = success or failure
            self.status.update(done=done, success=success, failure=failure)

            # 获得奖励
            reward += self.get_reward()

            if done:
                break

        reward = reward / (sub_step + 1)
        if hasattr(self, 'logger'):
            self.logger.store_buffer(action=action, reward=reward)

        return self.get_observation(), reward, done, other_info
