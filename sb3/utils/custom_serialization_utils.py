#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""保存配置到json

来源自spinup/utils/serialization_utils，在log中负责将当前所有变量转换到json并保存
嫌弃对ndarray兼容不好，自己改
递归转换

效果很好，但是文件也变大了

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
1/29/23 4:05 PM   yinzikang      1.0         None
"""
import json
import numpy


def convert_json(obj):
    """ Convert obj to a version which can be serialized with JSON. """
    """复杂的类型不会经过任何支路，直接输出str(obj)
        很多类型如类、方法都是有默认的str的，所以不用操心，只需要把自己想特别处理的加上即可
    """
    if is_json_serializable(obj):
        return obj
    else:
        if isinstance(obj, dict):
            return {convert_json(k): convert_json(v)
                    for k, v in obj.items()}

        elif isinstance(obj, tuple):
            return (convert_json(x) for x in obj)

        elif isinstance(obj, list):
            return [convert_json(x) for x in obj]

        elif isinstance(obj, numpy.ndarray):
            return convert_json(obj.tolist())

        elif hasattr(obj, '__name__') and not ('lambda' in obj.__name__):
            return convert_json(obj.__name__)

        elif hasattr(obj, '__dict__') and obj.__dict__:
            obj_dict = {convert_json(k): convert_json(v)
                        for k, v in obj.__dict__.items()}
            return {str(obj): obj_dict}

        return str(obj)


def is_json_serializable(v):
    try:
        json.dumps(v)
        return True
    except:
        return False
