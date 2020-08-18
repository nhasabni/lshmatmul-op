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

#define PADDING 4

#define BISINIT 0
#define BINDEX 1
#define BCOUNT 2


    int Bucket_add(int id, int l, int idx, int L, int RangePow, int BUCKETSIZE,
                   double *lsh_table) {

        double &_count = lsh_table[l * (RangePow << 1) * (BUCKETSIZE + PADDING)
			 + idx * (BUCKETSIZE + PADDING) + BUCKETSIZE + BCOUNT];
        double &_isInit = lsh_table[l * (RangePow << 1) * (BUCKETSIZE + PADDING)
                          + idx * (BUCKETSIZE + PADDING) + BUCKETSIZE + BISINIT];
        double &_index = lsh_table[l * (RangePow << 1) * (BUCKETSIZE + PADDING)
                         + idx * (BUCKETSIZE + PADDING) + BUCKETSIZE + BINDEX];
    //    if (FIFO) {
            _isInit += 1; // local to bucket, not used yet
            int index = _counts & (BUCKETSIZE - 1);
            _counts++; // local to each layer
            lsh_table[l * (RangePow << 1) * (BUCKETSIZE + PADDING) +
                idx * (BUCKETSIZE + PADDING)  + index] = id;
            return index;
    //    }
     }

    int* LSH_add(int *indices, int id, int _L)
    {
        int * secondIndices = new int[_L];
        for (int i = 0; i < _L; i++)
        {
            //secondIndices[i] = _bucket[i][indices[i]].add(id);
            secondIndices[i] = Bucket_add(id, i, indices[i], _L, RangePow,
                                          BUCKETSIZE, lsh_table);
        }

        return secondIndices;
    }

    int* LSH_hashesToIndex(int * hashes, int _L, int _K)
    {

        int * indices = new int[_L];
        for (int i = 0; i < _L; i++)
        {
            unsigned int index = 0;

            for (int j = 0; j < _K; j++)
            {

                if (HashFunction==4){
                    unsigned int h = hashes[_K*i + j];
                    index += h<<(_K-1-j);
                }
//              else if (HashFunction==1 | HashFunction==2){
//                  unsigned int h = hashes[_K*i + j];
//                  index += h<<((_K-1-j)*(int)floor(log(binsize)));
//                  }
//                  else {
//                    unsigned int h = rand1[_K*i + j];
//                    h *= rand1[_K * i + j];
//                    h ^= h >> 13;
//                    h ^= rand1[_K * i + j];
//                    index += h * hashes[_K * i + j];
//                  }
            }
//          if (HashFunction==3) {
//            index = index&((1<<_RangePow)-1);
//          }
            indices[i] = index;
        }

        return indices;
    }

    // Needs to be modified
    int * Bucket_getAll(int l, int idx, int L, int RangePow, int BUCKETSIZE,
                        double *lsh_table) {

        double &_count = lsh_table[l * (RangePow << 1) * (BUCKETSIZE + PADDING)
                         + idx * (BUCKETSIZE + PADDING) + BUCKETSIZE + BCOUNT];
        double &_isInit = lsh_table[l * (RangePow << 1) * (BUCKETSIZE + PADDING)
                          + idx * (BUCKETSIZE + PADDING) + BUCKETSIZE + BISINIT];
        double &_index = lsh_table[l * (RangePow << 1) * (BUCKETSIZE + PADDING)
                         + idx * (BUCKETSIZE + PADDING) + BUCKETSIZE + BINDEX];

        if (_isInit == -1)
            return NULL;
        if(_count<BUCKETSIZE) {
          lsh_table[l * (RangePow << 1) * (BUCKETSIZE + PADDING)
          + idx * (BUCKETSIZE + PADDING)  + _count] = -1;
        }
        return lsh_table;
    }

    /*
    * Returns all the buckets
    */
    int** LSH_retrieveRaw(int *indices, int _L, int RangePow, int BUCKETSIZE,
                          double * lsh_table) {
        int ** rawResults = new int*[_L];
        int count = 0;

        for (int i = 0; i < _L; i++) {
          // update to tensor part
          // rawResults[i] = _bucket[i][indices[i]].getAll();
          rawResults[i] = Bucket_getAll(i, indices[i], _L, RangePow, BUCKETSIZE, lsh_table);
        }
        return rawResults;
    }

  int* getHash(float *vector, int length, int **_indices, int **_randBits,
               int _numhashes) {

    int *hashes = new int[_numhashes];
    _samSize = ceil(1.0*vector / Ratio);
// #pragma omp parallel for
    for (int i = 0; i < _numhashes; i++) {
      double s = 0;
      for (size_t j = 0; j < _samSize; j++) {
        float v = vector[_indices[i][j]];
        if (_randBits[i][j] >= 0) {
          s += v;
        }
        else {
          s -= v;
        }
      }

    hashes[i] = (s >= 0 ? 0 : 1);
    }
    return hashes;
  }


    int* getHashSparse(int* indices, float *values, int length, int **_indices,
                       int **_randBits, int _numhashes) {
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
              }
              else {
                s -= v;
              }
              i++;
              j++;
            }
            else if (indices[i] < _indices[p][j]){
              i++;
            }
            else {
              j++;
            }
          }
          hashes[p] = (s >= 0 ? 0 : 1);
       }

       return hashes;
    }

  int queryActiveNodeandComputeActivations(int** activenodesperlayer,
                                           float** activeValuesperlayer,
                                           int* lengths, int layerIndex,
                                           int inputID, int* label,
                                           int labelsize, float Sparsity,
                                           int iter) {
    int len;
    int in = 0;

    if(Sparsity == 1.0) {
      len = _noOfNodes;
      lengths[layerIndex + 1] = len;
      activenodesperlayer[layerIndex + 1] = new int[len]; //assuming not intitialized;
      for (int i = 0; i < len; i++) {
        activenodesperlayer[layerIndex + 1][i] = i;
      }
    }
    else {
      int *hashes;
      hashes = getHashSparse(activenodesperlayer[layerIndex],
                             activeValuesperlayer[layerIndex],
                             lengths[layerIndex], _indices,
                             _randBits, _numhashes);
      int *hashIndices = LSH_hashesToIndex(hashes, _L, _K);
      int **actives = LSH_retrieveRaw(hashIndices, _L, RangePow, BUCKETSIZE,
                                      lsh_table);
      // we now have a sparse array of indices of active nodes

      // Get candidates from hashtable
      std::map<int, size_t> counts;
      // Make sure that the true label node is in candidates
      if (_type == NodeType::Softmax && labelsize > 0) {
        for (int i = 0; i < labelsize ;i++){
          counts[label[i]] = _L;
        }
      }

      for (int i = 0; i < _L; i++) {
        if (actives[i] == NULL) {
          continue;
        } 
        else {
          // copy sparse array into (dense) map
          for (int j = 0; j < BUCKETSIZE; j++) {
            int tempID = actives[i][j] - 1;
            if (tempID >= 0) {
              counts[tempID] += 1;
            }
            else {
              break;
            }
          }
        }
      }
      // needs to be modified
/*
            in = counts.size();
            if (counts.size()<1500){
                srand(time(NULL));
                int start = rand() % _noOfNodes;
                for (int i = start; i < _noOfNodes; i++) {
                    if (counts.size() >= 1000) {
                        break;
                    }
                    if (counts.count(_randNode[i]) == 0) {
                        counts[_randNode[i]] = 0;
                    }
                }

                if (counts.size() < 1000) {
                    for (int i = 0; i < _noOfNodes; i++) {
                        if (counts.size() >= 1000) {
                            break;
                        }
                        if (counts.count(_randNode[i]) == 0) {
                            counts[_randNode[i]] = 0;
                        }
                    }
                }
            }

            len = counts.size();
            lengths[layerIndex + 1] = len;
            activenodesperlayer[layerIndex + 1] = new int[len];

            // copy map into new array
            int i=0;
            for (auto &&x : counts) {
                activenodesperlayer[layerIndex + 1][i] = x.first;
                i++;
            }

*/

      delete[] hashes;
      delete[] hashIndices;
      delete[] actives;

    }

    //assuming its not initialized else memory leak;
    activeValuesperlayer[layerIndex + 1] = new float[len];
    float maxValue = 0;
    if (_type == NodeType::Softmax)
      _normalizationConstants[inputID] = 0;

    // find activation for all ACTIVE nodes in layer
    for (int i = 0; i < len; i++) {
      activeValuesperlayer[layerIndex + 1][i] = 
	_Nodes[activenodesperlayer[layerIndex + 1][i]].getActivation(
	activenodesperlayer[layerIndex], activeValuesperlayer[layerIndex],
        lengths[layerIndex], inputID);

      if(_type == NodeType::Softmax && activeValuesperlayer[layerIndex + 1][i] >
	 maxValue){
         maxValue = activeValuesperlayer[layerIndex + 1][i];
      }
    }

    if(_type == NodeType::Softmax) {
      for (int i = 0; i < len; i++) {
        float realActivation = exp(activeValuesperlayer[layerIndex + 1][i] -
                                   maxValue);
        activeValuesperlayer[layerIndex + 1][i] = realActivation;
        _Nodes[activenodesperlayer[layerIndex + 1][i]].SetlastActivation(
                                                       inputID, realActivation);
        _normalizationConstants[inputID] += realActivation;
      }
    }

    return in;
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

