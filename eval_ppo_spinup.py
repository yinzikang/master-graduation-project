#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Write target here

Write detailed description here

Write typical usage example here

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
4/9/23 10:26 PM   yinzikang      1.0         None
"""
import pandas
import matplotlib.pyplot as plt
path = "./spinup/2023-04-09_ppo/2023-04-09_22-31-11-ppo_s7"
result_txt = pandas.read_table(path + '/progress.txt', sep='\t', engine='python')

for idx, (name, series) in enumerate(result_txt.items()):
    plt.figure(idx + 1)
    plt.plot(series.to_numpy())
    plt.title(name)
plt.show()