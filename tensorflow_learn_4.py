#! -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]
print(x_data.shape)
noise = np.random.normal(0, 0.02, 10)
print(noise)
