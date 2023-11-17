#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time         : 2023/11/17 13:29
# @Author       : adolf
# @Email        : adolf1321794021@gmail.com
# @LastEditTime : 2023/11/17 13:29
# @File         : trainer.py
# initialise the LightningModule
import os
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from monai_lightning.model import Net

net = Net()


# set up loggers and checkpoints
log_dir = "logs/"
tb_logger = TensorBoardLogger(save_dir=log_dir)

# initialise Lightning's trainer.
trainer = Trainer(
    devices=[0],
    max_epochs=600,
    logger=tb_logger,
    enable_checkpointing=True,
    num_sanity_val_steps=1,
    log_every_n_steps=16,
)

# train
trainer.fit(net)
