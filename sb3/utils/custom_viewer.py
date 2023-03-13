#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Write target here

Write detailed description here

Write typical usage example here

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
1/26/23 10:18 PM   yinzikang      1.0         None
"""
from utils.mujoco_viewer.mujoco_viewer import MujocoViewer
import mujoco as mp


class EnvViewer(MujocoViewer):
    def __init__(self, env, pause_start = False):
        super().__init__(env.mjc_model, env.data, hide_menus=False)
        self.vopt.flags[mp.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        self.vopt.flags[mp.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        self.vopt.flags[mp.mjtVisFlag.mjVIS_CONTACTSPLIT] = True
        self._paused = pause_start

    def viewer_render(self):
        self.render()

    # def __del__(self):
    #     self.close()
