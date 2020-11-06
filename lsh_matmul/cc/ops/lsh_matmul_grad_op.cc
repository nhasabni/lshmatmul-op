#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"


using namespace tensorflow;

// the gradients are simply passed as additional arguments as
// they are available in the Python function for registering the gradient operation.
REGISTER_OP("LshMatmulGrad")
  .Input("grad: float32")
  .Input("input: float32")
  .Input("weights: float32")
  .Output("grad_input: float32")
  .Output("grad_weights: float32");