from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.keras.layers import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.ops import nn

try:
  from tensorflow_lsh_matmul.python.ops.lsh_matmul_op import lsh_matmul
except ImportError:
  from lsh_matmul_op import lsh_matmul
from tensorflow_lsh_matmul.python.keras.layers.initializers import *

class LSHMatMulLayer(Layer):
  """Just your regular densely-connected NN layer.

  `Dense` implements the operation:
  `output = activation(dot(input, kernel) + bias)`
  where `activation` is the element-wise activation function
  passed as the `activation` argument, `kernel` is a weights matrix
  created by the layer, and `bias` is a bias vector created by the layer
  (only applicable if `use_bias` is `True`).

  Note: if the input to the layer has a rank greater than 2, then
  it is flattened prior to the initial dot product with `kernel`.

  Example:

  ```python
      # as first layer in a sequential model:
      model = Sequential()
      model.add(Dense(32, input_shape=(16,)))
      # now the model will take as input arrays of shape (*, 16)
      # and output arrays of shape (*, 32)

      # after the first layer, you don't need to specify
      # the size of the input anymore:
      model.add(Dense(32))
  ```

  Arguments:
      units: Positive integer, dimensionality of the output space.
      activation: Activation function to use.
          If you don't specify anything, no activation is applied
          (ie. "linear" activation: `a(x) = x`).
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix.
      bias_initializer: Initializer for the bias vector.
      kernel_regularizer: Regularizer function applied to
          the `kernel` weights matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      activity_regularizer: Regularizer function applied to
          the output of the layer (its "activation")..
      kernel_constraint: Constraint function applied to
          the `kernel` weights matrix.
      bias_constraint: Constraint function applied to the bias vector.

  Input shape:
      nD tensor with shape: `(batch_size, ..., input_dim)`.
      The most common situation would be
      a 2D input with shape `(batch_size, input_dim)`.

  Output shape:
      nD tensor with shape: `(batch_size, ..., units)`.
      For instance, for a 2D input with shape `(batch_size, input_dim)`,
      the output would have shape `(batch_size, units)`.
  """

  def __init__(self,
               units,
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               bucketsize=128,
               K=1,
               L=1,
               **kwargs):
    if 'input_shape' not in kwargs and 'input_dim' in kwargs:
      kwargs['input_shape'] = (kwargs.pop('input_dim'),)

    super(LSHMatMulLayer, self).__init__(
        activity_regularizer=regularizers.get(activity_regularizer), **kwargs)
    self.units = int(units)
    self.activation = activations.get(activation)
    self.use_bias = use_bias
    self.kernel_initializer = initializers.get(kernel_initializer)
    self.bias_initializer = initializers.get(bias_initializer)
    self.srp_rand_bits_initializer = SRPRandBitsInitializer
    self.srp_indices_initializer = SRPIndicesInitializer
    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)
    self.kernel_constraint = constraints.get(kernel_constraint)
    self.bias_constraint = constraints.get(bias_constraint)
    self.L=int(L)
    self.K=int(K)
    self.bucketsize=int(bucketsize)
    self.supports_masking = True
    self.input_spec = InputSpec(min_ndim=2)


  def build(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    if tensor_shape.dimension_value(input_shape[-1]) is None:
      raise ValueError('The last dimension of the inputs to `Dense` '
                       'should be defined. Found `None`.')
    last_dim = tensor_shape.dimension_value(input_shape[-1])
    self.input_spec = InputSpec(min_ndim=2,
                                axes={-1: last_dim})
    self.kernel = self.add_weight(
        'kernel',
        shape=[last_dim, self.units],
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        dtype=self.dtype,
        trainable=True)
    if self.use_bias:
      self.bias = self.add_weight(
          'bias',
          shape=[1],
          initializer=self.bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          dtype=self.dtype,
          trainable=True)
    else:
      self.bias = None
    
    #allocate hash tables
    self.buckets = self.add_weight(
          'buckets',
          shape=[self.L, pow(2,self.K), self.bucketsize],
          initializer=self.bias_initializer, # 
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          dtype=dtypes.int32,
          trainable=True)
    self.randBits =  self.add_weight(
          'randbits',
          shape=[self.L, self.K, last_dim],
          initializer=self.srp_rand_bits_initializer, # random odd numbers/initializer
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          dtype=dtypes.int16,
          trainable=True)    
    self.indices =  self.add_weight(
          'indices',
          shape=[self.L, self.K, last_dim],
          initializer=self.srp_indices_initializer, # random odd numbers
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          dtype=dtypes.int32,
          trainable=True)    
   
    self.built = True

  def call(self, inputs):
    inputs = ops.convert_to_tensor(inputs)

    #outputs = gen_user_ops.mat_mul(inputs, self.kernel)
    outputs = lsh_matmul(inputs, self.buckets, self.indices, self.randBits, self.kernel)
    if self.use_bias:
      outputs = nn.bias_add(outputs, self.bias)
    if self.activation is not None:
      return self.activation(outputs)  # pylint: disable=not-callable
    return outputs

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    input_shape = input_shape.with_rank_at_least(2)
    if tensor_shape.dimension_value(input_shape[-1]) is None:
      raise ValueError(
          'The innermost dimension of input_shape must be defined, but saw: %s'
          % input_shape)
    return input_shape[:-1].concatenate(self.units)

  def get_config(self):
    config = {
        'units': self.units,
        'activation': activations.serialize(self.activation),
        'use_bias': self.use_bias,
        'kernel_initializer': initializers.serialize(self.kernel_initializer),
        'bias_initializer': initializers.serialize(self.bias_initializer),
        'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
        'bias_regularizer': regularizers.serialize(self.bias_regularizer),
        'activity_regularizer':
            regularizers.serialize(self.activity_regularizer),
        'kernel_constraint': constraints.serialize(self.kernel_constraint),
        'bias_constraint': constraints.serialize(self.bias_constraint)
    }
    base_config = super(LSHMatMulLayer, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

