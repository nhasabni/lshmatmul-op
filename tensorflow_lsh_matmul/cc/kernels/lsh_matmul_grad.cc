#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;


/// \brief Implementation of an inner product gradient operation.
/// Note that this operation is used in Python to register the gradient as
/// this is not possible in C*+ right now.
/// \param context
class lshMatmulGradOp : public OpKernel {
public:
  /// \brief Constructor.
  /// \param context
  explicit lshMatmulGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  /// \brief Compute the inner product gradients.
  /// \param context
  void Compute(OpKernelContext* context) override {
    
    // output and grad is provided as input
    DCHECK_EQ(3, context->num_inputs());

    // get the gradient tensor
    const Tensor& grad = context->input(0);
    
    // get the original input tensor
    const Tensor& input = context->input(1);
    
    // get the weight tensor
    const Tensor& weights = context->input(2);
    
    // create input shape (inferred from the additional attribute `n`)
    TensorShape input_shape = input.shape();
    TensorShape weights_shape = weights.shape();
    
    DCHECK_EQ(input_shape.dim_size(0), weights_shape.dim_size(1)); //input = KM
    DCHECK_EQ(weights_shape.dim_size(0), grad.shape().dim_size(0)); // weights = NK
    
    // create output tensors
    Tensor* grad_input = NULL;
    Tensor* grad_weights = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_shape, &grad_input));
    OP_REQUIRES_OK(context, context->allocate_output(1, weights_shape, &grad_weights));
    
    // get the Eigen tensors for data access
    auto grad_tensor = grad.matrix<float>();
    auto weights_tensor = weights.matrix<float>();
    auto input_tensor = input.matrix<float>();
    auto grad_input_tensor = grad_input->matrix<float>();
    auto grad_weights_tensor = grad_weights->matrix<float>();
    
    // doing it manually for ismplicity
    for (int i = 0; i < weights_shape.dim_size(0); i++) { //K -> MN x (KN)T 
      grad_input_tensor(i, 0) = 0;
      for (int j = 0; j < grad.shape().dim_size(0); j++) { //N
        grad_input_tensor(i, 0) += grad_tensor(j, 0)*weights_tensor(j, i);
      }
    }
    
    
    for (int i = 0; i < weights_shape.dim_size(0); i++) { //K -> (MK)T x MN
      for (int j = 0; j < weights_shape.dim_size(1); j++) { //M
        grad_weights_tensor(i, j) = grad_tensor(i, 0)*input_tensor(j, 0);
      }
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("LshMatmulGrad").Device(DEVICE_CPU), lshMatmulGradOp);
