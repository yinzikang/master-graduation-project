#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Write target here

Write detailed description here

Write typical usage example here

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
1/26/23 10:18 PM   yinzikang      1.0         None
"""
from gym_custom.utils.mujoco_viewer.mujoco_viewer import MujocoViewer
import mujoco as mp


class EnvViewer(MujocoViewer):
    def __init__(self, env, pause_start = False, view_force=True):
        super().__init__(env.mjc_model, env.data, hide_menus=False)
        # self.vopt.flags[mp.mjtVisFlag.mjVIS_CONTACTPOINT] = view_force
        self.vopt.flags[mp.mjtVisFlag.mjVIS_CONTACTFORCE] = view_force
        self.vopt.flags[mp.mjtVisFlag.mjVIS_CONTACTSPLIT] = view_force
        self.vopt.flags[mp.mjtVisFlag.mjVIS_CONSTRAINT] = False
        self._paused = pause_start

    def viewer_render(self):
        self.render()

    # def __del__(self):
    #     self.close()
