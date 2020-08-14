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

using namespace tensorflow;

typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename Device, typename T, bool USE_CUBLAS>

class MklMatMulOp : public OpKernel {
 public:
  explicit MklMatMulOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_a", &transpose_a_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_b", &transpose_b_));
  }

  int* getHash(float vector, int length, int _indices, int _randBits, int _numhashes) {
    // length should be = to _dim
        int *hashes = new int[_numhashes];
        _samSize = ceil(1.0*vector / Ratio);
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

void Compute(OpKernelContext* ctx) override {
    const Tensor& a = ctx->input(0);
    const Tensor& b = ctx->input(1);

    // Tensor v = ctx->mutable_input(2, true);
    Tensor v1 = ctx->mutable_input(2, true);
    Tensor v2 = ctx->mutable_input(3, true);
    Tensor v3 = ctx->mutable_input(4, true);
    Tensor v4 = ctx->mutable_input(5, true);
    // const Tensor& v = ctx->input(2);
	

    //	auto v_arr = v.flat<float>(); // command to access and print a tensor data
    //	const int N = v_arr.size();
    //	for (int i = 0; i < N; i++)
    //		printf("%lf ", v_arr(i));
    //	v_arr(0)++;

    printf("-----------------Zafar Jul-8 MKL// in compute-------------------\n");
    // return ;
    // Check that the dimensions of the two matrices are valid.
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(a.shape()),
                errors::InvalidArgument("In[0] is not a matrix"));
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(b.shape()),
                errors::InvalidArgument("In[1] is not a matrix"));
    Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair;
    dim_pair[0].first = transpose_a_ ? 0 : 1;
    dim_pair[0].second = transpose_b_ ? 1 : 0;

    OP_REQUIRES(
        ctx, a.dim_size(dim_pair[0].first) == b.dim_size(dim_pair[0].second),
        errors::InvalidArgument(
            "Matrix size-incompatible: In[0]: ", a.shape().DebugString(),
            ", In[1]: ", b.shape().DebugString()));
    int a_dim_remaining = 1 - dim_pair[0].first;
    int b_dim_remaining = 1 - dim_pair[0].second;
    TensorShape out_shape(
        {a.dim_size(a_dim_remaining), b.dim_size(b_dim_remaining)});
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &out));
/*	
	int K, L;
	K = 128;
	L = 1;
	TensorShape lsh_shape(
        {L, K});
	Tensor* lsh = nullptr;
	OP_REQUIRES_OK(ctx, ctx->allocate_output(1, lsh_shape, &lsh));
*/
    if (out->NumElements() == 0) {
      // If a has shape [0, x] or b has shape [x, 0], the output shape
      // is a 0-element matrix, so there is nothing to do.
      return;
    }

    if (a.NumElements() == 0 || b.NumElements() == 0) {
      // If a has shape [x, 0] and b has shape [0, y], the
      // output shape is [x, y] where x and y are non-zero, so we fill
      // the output with zeros.
      functor::SetZeroFunctor<Device, T> f;
      f(ctx->eigen_device<Device>(), out->flat<T>());
      return;
    }

    const int m = a.dim_size(1 - dim_pair[0].first);
    const int k = a.dim_size(dim_pair[0].first);
    const int n = b.dim_size(1 - dim_pair[0].second);
    bool transpose_a = dim_pair[0].first == 0;
    bool transpose_b = dim_pair[0].second == 1;

    auto a_ptr = (a.template flat<T>().data());
    auto b_ptr = (b.template flat<T>().data());
    auto c_ptr = (out->template flat<T>().data());
	
/*
    K = 2
    L = 20    RangePow = 6
    Sparsity = 1
    NumberofNodes = 128    
    PrevNumNode = 135909 # for fisrt layer, this is input dim
    BUCKETSIZE = 128
   self.bucket_arr = self.add_weight(
        'Sbkt',
        shape=[L, RangePow, (BUCKETSIZE + 4)]

    self.randBits = self.add_weight(
        'SlRnd',
        shape=[_numhashes, _samSize],

    self.indices = self.add_weight(
      'SlInd',
      shape=[_numhashes, _samSize],

    self.weight = self.add_weight(
        'SlW',
        shape=[32,],


*/
//	auto v_ptr = (v.template flat<T>().data());

    auto bucket_arr = (v1.template flat<T>().data());
    auto  weights= (v4.template flat<T>().data());
    auto  _indices = (v3.template flat<T>().data());
    auto  _randBits = (v2.template flat<T>().data());
    //	for (int i = 0; i < 5; i++){
    //		std::cout << v_ptr[i] << " ";
    //	}
    //	std::cout << std::endl;
    std::cout << bucket_arr << " ";
    std::cout << std::endl;  

    _numhashes = 32;
    _randBits =  new short [_numhashes];
    _indices = new int [_numhashes];
    getHash(weights, _numhashes, _randBits, _indices, _numhashes);
 
    MklBlasGemm(transpose_a, transpose_b, m, n, k, a_ptr, transpose_a ? m : k,
             b_ptr, transpose_b ? k : n, c_ptr, n);
    
  }



};

#define REGISTER_CPU(T)                                         \
  REGISTER_KERNEL_BUILDER(                                      \
      Name("LshMatMul").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      MklMatMulOp<CPUDevice, T, false /* cublas, ignored for CPU */>);

#ifdef ENABLE_MKL
TF_CALL_float(REGISTER_CPU);

#ifndef INTEL_MKL_DNN_ONLY
TF_CALL_double(REGISTER_CPU);
TF_CALL_complex64(REGISTER_CPU);
TF_CALL_complex128(REGISTER_CPU);
#endif  // !INTEL_MKL_DNN_ONLY
#endif  // ENABLE_MKL

