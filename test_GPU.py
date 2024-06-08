import os
os.environ["CUDA_VISIBLE_DEVICES"] = "GPU device number"

import tensorflow as tf
# Check if GPU is available
if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")

