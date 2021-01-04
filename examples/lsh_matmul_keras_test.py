import tensorflow as tf
from tensorflow_lsh_matmul.python.keras.layers import LSHMatMulLayer


# MxK = 1x2, x KxN=2x2 w
M=2
K=2
N=2
lsh_matmul_layer = LSHMatMulLayer(N, input_shape=(M, K), bucketsize=32, K=2, L=2)
print(lsh_matmul_layer(tf.ones([M, K])))
''' print output shape '''
print(lsh_matmul_layer.compute_output_shape([M, K]))

print(lsh_matmul_layer.get_config())

