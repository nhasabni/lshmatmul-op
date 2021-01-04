from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import random
import time

from tensorflow.python.ops.init_ops import Initializer  # pylint: disable=unused-import
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops

class SRPRandBitsInitializer(Initializer):

  def __init__(self, dtype=dtypes.int16):
    self.dtype = dtypes.as_dtype(dtype)

  def __call__(self, shape, dtype=None, partition_info=None):
    if dtype is None:
      dtype = self.dtype
    # Randbits shape is: [L, K, last_dim]
    last_dim = shape[-1]
    num_hashes = shape[0] * shape[1]
    ratio = 1
    sam_size = math.ceil(1.0 * last_dim / ratio)
    
    seed = int(time.time())
    curr = random_ops.random_uniform([num_hashes, sam_size], maxval=2147483647, 
                                     seed=seed, dtype=dtypes.int32)
    # Get a Numpy array initialized to 0 but with first two dimensions flattened.
    # We use Numpy array because Tensors in Tensorflow are immutable.
    # We later convert this array to Tensor.
    rand_bits = np.zeros([num_hashes, sam_size], dtype=np.int16)
    for i in range(num_hashes):
      for j in range(sam_size):
        if curr[i][j] % 2 == 0:
          rand_bits[i][j] = 1
        else:
          rand_bits[i][j] = -1
    rand_bits_tensor = ops.convert_to_tensor(rand_bits)
    return array_ops.reshape(rand_bits_tensor, shape)



class SRPIndicesInitializer(Initializer):

  def __init__(self, dtype=dtypes.int32):
    self.dtype = dtypes.as_dtype(dtype)

  def __call__(self, shape, dtype=None, partition_info=None):
    if dtype is None:
      dtype = self.dtype
    # Randbits shape is: [L, K, last_dim]
    last_dim = shape[-1]
    num_hashes = shape[0] * shape[1]
    ratio = 1
    sam_size = math.ceil(1.0 * last_dim / ratio)
    a = list(range(last_dim))
    
    # Get a Numpy array initialized to 0 but with first two dimensions flattened.
    # We use Numpy array because Tensors in Tensorflow are immutable.
    # We later convert this array to Tensor.
    indices = np.zeros([num_hashes, sam_size], dtype=np.int32)
    for i in range(num_hashes):
      random.shuffle(a)
      for j in range(sam_size):
        indices[i][j] = a[j]
      indices[i].sort()
    indices_tensor = ops.convert_to_tensor(indices)
    return array_ops.reshape(indices_tensor, shape)
