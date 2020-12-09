import tensorflow as tf
from tensorflow_lsh_matmul.python.keras.layers import LSHMatMulLayer


# MxK = 1x2, KxN=2x2
M=1
K=1
N=1
lsh_matmul_layer = LSHMatMulLayer(N, input_shape=(M, K), bucketsize=32, K=1, L=1)
print(lsh_matmul_layer(tf.ones([M, K])))
''' print output shape '''
print(lsh_matmul_layer.compute_output_shape([M, K]))

print(lsh_matmul_layer.get_config())
