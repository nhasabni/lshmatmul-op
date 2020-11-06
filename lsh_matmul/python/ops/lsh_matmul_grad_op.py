#!/usr/bin/env python3
"""
Gradients for inner product.
"""

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops
inner_product_grad_module = tf.load_op_library('./lsh_matmul_grad.so')

@ops.RegisterGradient("LshMatmul")
#def _lsh_matmul_grad_cc(op, grad):
def lsh_matmul_grad(op, grad):
    """
    The gradient for `lsh_matmul` using the operation implemented in C++.
    
    :param op: `lsh_matmul` `Operation` that we are differentiating, which we can use
        to find the inputs and outputs of the original op.
    :param grad: gradient with respect to the output of the `lsh_matmul` op.
    :return: gradients with respect to the input of `lsh_matmul`.
    """
    
    return lsh_matmul_grad_module.lsh_matmul_grad(grad, op.inputs[0], op.inputs[1])

# uncomment this and comment the corresponding line above to use the Python
# implementation of the inner product gradient
#
#@ops.RegisterGradient("LshMatmul")
'''
def _lsh_matmul_grad(op, grad):
    """
    The gradients for `lsh_matmul`.
    
    :param op: `lsh_matmul` `Operation` that we are differentiating, which we can use
        to find the inputs and outputs of the original op.
    :param grad: gradient with respect to the output of the `lsh_matmul` op.
    :return: gradients with respect to the input of `lsh_matmul`.
    """
  
    input_tensor = op.inputs[0]
    weight_tensor = op.inputs[1]
    input_rows = array_ops.shape(input_tensor)[0]
    output_rows = array_ops.shape(weight_tensor)[0]
    
    grad_input = tf.matmul(tf.transpose(grad), weight_tensor)
    grad_weights = tf.multiply(tf.transpose(grad), tf.reshape(tf.tile(tf.reshape(input_tensor, [input_rows]), [output_rows]), [output_rows, -1]))
    
    return [tf.transpose(grad_input), grad_weights]
    '''