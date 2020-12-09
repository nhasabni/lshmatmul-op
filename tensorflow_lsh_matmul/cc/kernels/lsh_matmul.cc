
/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

typedef Eigen::ThreadPoolDevice CPUDevice;

//template <typename Device, typename T, bool USE_CUBLAS>

/*
class MklMatMulOp : public OpKernel {
 public:
  explicit MklMatMulOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_a", &transpose_a_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_b", &transpose_b_));
  }

#define PADDING 4
#define BISINIT 0
#define BINDEX 1
#define BCOUNT 2
//TODO: fill the values below
#define BUCKETSIZE 
#define _L
#define _numhashes 
#define _noOfNodes 

  int Bucket_add(int id, int l, int idx, int L, int RangePow, int BUCKETSIZE,
                 double *lsh_table) {

  int getHashSparse(int* indices, float *values, int length) {
    int *hashes = new int[_numhashes];

    for (int p = 0; p < _numhashes; p++) {
        double s = 0;
        int i = 0;
        int j = 0;
        while (i < length & j < _samSize) {
          if (indices[i] == _indices[p][j]) {
            float v = values[i];
            if (_randBits[p][j] >= 0) {
              s += v;
            } else {
              s -= v;
            }
            i++;
            j++;
          }
          else if (indices[i] < _indices[p][j]){
            i++;
          }
          else{
            j++;
          }
        }
        hashes[p] = (s >= 0 ? 0 : 1);
    }
    return rawResults;
  }

    return hashes;
  }

  int* hashesToIndex(int * hashes) {
    int * indices = new int[_L];
    for (int i = 0; i < _L; i++) {
      unsigned int index = 0;
      for (int j = 0; j < _K; j++) {
        if (HashFunction==4) {
	  unsigned int h = hashes[_K*i + j];
	  index += h<<(_K-1-j);
	} 
      indices[i] = index;
      }
    } 
    return indices;
  }

  int add(int tableId, int indices, int id) {
	int secondIndices = _bucket[tableId][indices].add(id);
	return secondIndices;
  }

  int getHash(float *vector, int length) {
    // length should be = to _dim
    int *hashes = new int[_numhashes];

 // #pragma omp parallel for
    for (int i = 0; i < _numhashes; i++) {
        double s = 0;
        for (size_t j = 0; j < _samSize; j++) {
            float v = vector[_indices[i][j]];
            if (_randBits[i][j] >= 0) {
                s += v;
            } else {
                s -= v;
            }
        }
        hashes[i] = (s >= 0 ? 0 : 1);
    }
    return hashes;
}

  int** retrieveRaw(int *indices) {
    int ** rawResults = new int*[_L];
    int count = 0;
    for (int i = 0; i < _L; i++)
      rawResults[i] = _bucket[i][indices[i]].getAll();
    return rawResults;
  }

  int getCandidates(int **actives) {
    for(int i = 0; i < _L; i++) {
      if(actives[i] == NULL) {
        continue;
      } 
      else {
        for(int j = 0; j < BUCKETSIZE; j++) {
          int tempID = actives[i][j] - 1;
          if(tempID >= 0) 
            counts[tempID] += 1;
          else
            break;
        }
      }
    }

    if(counts.size()<1500){
      srand(time(NULL));
      int start = rand() % _noOfNodes;
      for(int i = start; i < _noOfNodes; i++) {
        if(counts.size() >= 1000) 
          break;
        if(counts.count(_randNode[i]) == 0)
          counts[_randNode[i]] = 0;
      }


      if(counts.size() < 1000) {
        for(int i = 0; i < _noOfNodes; i++) {
          if(counts.size() >= 1000)
            break;
          if(counts.count(_randNode[i]) == 0)
            counts[_randNode[i]] = 0;
        }
      }
    }

    len = counts.size();
    return len;
  }
};

void Compute(OpKernelContext* ctx) override {
    // Note:
    // need values(x), weights, indices, inputID,  activenodesPerLayer, 
    // activeValues & length as input parameters or attributes
    const Tensor& a = ctx->input(0);
    Tensor an = ctx->input(1);
    Tensor av = ctx->input(2);
    Tensor l = ctx->input(3);
    Tensor i = ctx->input(4);
    Tensor w = ctx->input(5);
    Tensor in = ctx->input(6);

    // Note: Not sure if we need transpose. 
    // TODO: check back later
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(a.shape()),
                errors::InvalidArgument("In[0] is not a matrix"));
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(w.shape()),
                errors::InvalidArgument("In[1] is not a matrix"));
    Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair;
    dim_pair[0].first = transpose_a_ ? 0 : 1;
    dim_pair[0].second = transpose_w_ ? 1 : 0;

    OP_REQUIRES(
        ctx, a.dim_size(dim_pair[0].first) == w.dim_size(dim_pair[0].second),
        errors::InvalidArgument(
            "Matrix size-incompatible: In[0]: ", a.shape().DebugString(),
            ", In[1]: ", w.shape().DebugString()));
    int a_dim_remaining = 1 - dim_pair[0].first;
    int w_dim_remaining = 1 - dim_pair[0].second;
    TensorShape out_shape(
        {a.dim_size(a_dim_remaining), w.dim_size(w_dim_remaining)});
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &out));
    if (out->NumElements() == 0) {
      // If a has shape [0, x] or b has shape [x, 0], the output shape
      // is a 0-element matrix, so there is nothing to do.
      return;
    }

    auto  weights = (w.template flat<T>().data());
    auto  indices = (i.template flat<T>().data());
    auto  inputID = (in.template flat<T>().data());
    auto  length = (l.template flat<T>().data());
    auto  activeNodesPerLayer = (an.template flat<T>().data());
    auto  activeValues = (av.template flat<T>().data());

    int *hashes;
    hashes = getHashSparse(activeNodesPerLayer, activeValues, length);
    int *hashIndices = _hashesToIndex(hashes);
    int **activeNodes = retrieveRaw(hashIndices);

    // get candidates from the list of active nodes (type 4-random selection)
    int len = getCandidates(actives); 
    // Matmul
    // y = Wx + b;
   
    // TODO: check for shape of out tensor 
    for(int i = 0; i < len; ++i) {
      out[inputID] = weights[indices[i]] * a[i];  
    }
    
  }

};
*/
/*
#define REGISTER_CPU(T)                                         \
  REGISTER_KERNEL_BUILDER(                                      \
      Name("LshMatMul").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      MklMatMulOp<CPUDevice, T, false >);
*/


/// \brief Implementation of an inner product operation.
/// \param context

/*
  .Input("buckets: int32")
  .Input("indices: int32")
  .Input("randBits: int16")
  .Input("weights: float") 
  */
class LshMatmulOp : public OpKernel {
public:
  /// \brief Constructor.
  /// \param context
  explicit LshMatmulOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  /// \brief Compute the inner product.
  /// \param context
  void Compute(OpKernelContext* context) override {
    
    // some checks to be sure ...
    DCHECK_EQ(5, context->num_inputs());
    
    // get the input tensor
    const Tensor& input = context->input(0);
    
    // buckets
    const Tensor& buckets = context->input(1);
    
    // indices
    const Tensor& indices = context->input(2);
    
    // randbits
    const Tensor& randBits = context->input(3);
    
    
    // get the weight tensor
    const Tensor& weights = context->input(4);
    
    // check shapes of input and weights
    const TensorShape& input_shape = input.shape();
    const TensorShape& weights_shape = weights.shape();
    
    // check input is a standing vector
    DCHECK_EQ(input_shape.dims(), 2);
    DCHECK_EQ(input_shape.dim_size(1), 1);
    
    // check weights is matrix of correct size
    DCHECK_EQ(weights_shape.dims(), 2);
    DCHECK_EQ(input_shape.dim_size(0), weights_shape.dim_size(1));
    
    // create output shape
    TensorShape output_shape;
    output_shape.AddDim(weights_shape.dim_size(0));
    output_shape.AddDim(1);
            
    // create output tensor
    Tensor* output = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    
    
    std::cout<<randBits.DebugString()<<std::endl;
    
    // get the corresponding Eigen tensors for data access
    auto input_tensor = input.matrix<float>();
    auto weights_tensor = weights.matrix<float>();
    auto output_tensor = output->matrix<float>();
    
    for (int i = 0; i < output->shape().dim_size(0); i++) {
      output_tensor(i, 0) = 0;
      for (int j = 0; j < weights.shape().dim_size(1); j++) {
        output_tensor(i, 0) += weights_tensor(i, j)*input_tensor(j, 0);
      }
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("LshMatmul").Device(DEVICE_CPU), LshMatmulOp);

#ifdef ENABLE_MKL
TF_CALL_float(REGISTER_CPU);

#ifndef INTEL_MKL_DNN_ONLY
TF_CALL_double(REGISTER_CPU);
TF_CALL_complex64(REGISTER_CPU);
TF_CALL_complex128(REGISTER_CPU);
#endif  // !INTEL_MKL_DNN_ONLY
#endif  // ENABLE_MKL

