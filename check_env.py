import sys
import tensorflow as tf
import pandas, PIL, matplotlib, sklearn

print("Python:", sys.version)
print("TensorFlow:", tf.__version__)
print("Pandas:", pandas.__version__)
print("Pillow:", PIL.__version__)
print("Matplotlib:", matplotlib.__version__)
print("Scikit-learn:", sklearn.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))