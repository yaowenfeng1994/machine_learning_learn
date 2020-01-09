import os
import warnings

import numpy as np
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

warnings.simplefilter(action='ignore', category=FutureWarning)

if __name__ == "__main__":
    execute = 2
    if execute == 1:
        g = tf.Graph()
        with g.as_default():
            pass