import os
import warnings

import imageio
import numpy as np
import tensorflow as tf

from artificial_neural_network import load_mnist

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

warnings.simplefilter(action='ignore', category=FutureWarning)


