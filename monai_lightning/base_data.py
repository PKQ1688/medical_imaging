#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time         : 2023/11/17 11:58
# @Author       : adolf
# @Email        : adolf1321794021@gmail.com
# @LastEditTime : 2023/11/17 11:58
# @File         : base_data.py
import os

from monai.apps import download_and_extract


def get_origin_data(root_dir="data/monai_spleen"):
    print(root_dir)

    resource = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar"
    md5 = "410d4a301da4e5b2f6f86ec3ddba524e"

    compressed_file = os.path.join(root_dir, "Task09_Spleen.tar")
    data_dir = os.path.join(root_dir, "Task09_Spleen")
    if not os.path.exists(data_dir):
        download_and_extract(resource, compressed_file, root_dir, md5)
