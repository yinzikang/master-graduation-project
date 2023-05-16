#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""所有实验阴影图

Write detailed description here

Write typical usage example here

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
5/15/23 10:57 AM   yinzikang      1.0         None
"""
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns # 导入模块

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10.5
plt.rcParams['lines.linewidth'] = 2.0

env_name = 'TrainEnvVariableStiffnessAndPostureAndSM_v2-v8'
# test_name = 'cabinet surface with plan v7'
# time_name_list = ['05-15-11-11','05-07-20-57','05-08-09-51','05-15-17-42','05-16-09-48']
test_name = 'cabinet drawer open with plan'
time_name_list = ['05-08-10-34', '05-15-10-18','05-16-01-12']
# test_name = 'cabinet door open with plan'
# time_name_list = ['05-15-01-19', '05-15-01-20']
print(env_name)
print(test_name)
rl_name = 'PPO'

folder_path_name = 'tensorboard_export/' + test_name + '/' + rl_name
full_path_list = [folder_path_name + '/run-' + file + '__1-tag-rollout_ep_rew_mean.csv' for file in time_name_list]

# 从CSV文件中加载数据
data = pd.concat([pd.read_csv(f) for f in full_path_list])

# 对所有数据按照Step进行分组，并计算每个Step对应的Value的平均和标准差
grouped_data = data.groupby('Step')['Value'].agg(['mean', 'std']).reset_index()

# 绘制带阴影的折线图
sns.set(style="darkgrid")
plt.plot(grouped_data['Step'], grouped_data['mean'])
plt.fill_between(grouped_data['Step'], grouped_data['mean'] - grouped_data['std'], grouped_data['mean'] + grouped_data['std'], alpha=0.15)
plt.xlabel('interaction steps')
plt.ylabel('average return')
plt.xlim([0, 1310720])
plt.savefig(folder_path_name + '/'+test_name+'rl_shadow.png', dpi=600, bbox_inches='tight')
plt.show()