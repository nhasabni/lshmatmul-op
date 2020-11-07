CXX := g++
NVCC := nvcc
PYTHON_BIN_PATH = python

#ZERO_OUT_SRCS = $(wildcard tensorflow_zero_out/cc/kernels/*.cc) $(wildcard tensorflow_zero_out/cc/ops/*.cc)

#LSH_MATMUL_SRCS = $(wildcard lsh_matmul/cc/kernels/*.cc) $(wildcard lsh_matmul/cc/ops/*.cc)
LSH_MATMUL_SRCS = $(wildcard tensorflow_lsh_matmul/cc/kernels/lsh_matmul.cc) $(wildcard tensorflow_lsh_matmul/cc/ops/lsh_matmul_op.cc)
LSH_MATMUL_GRAD_SRCS = $(wildcard tensorflow_lsh_matmul/cc/kernels/lsh_matmul_grad.cc) $(wildcard tensorflow_lsh_matmul/cc/ops/lsh_matmul_grad_op.cc)

TF_CFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

CFLAGS = ${TF_CFLAGS} -fPIC -O2 -std=c++11
LDFLAGS = -shared ${TF_LFLAGS}

#ZERO_OUT_TARGET_LIB = tensorflow_zero_out/python/ops/_zero_out_ops.so
LSH_MATMUL_TARGET_LIB = tensorflow_lsh_matmul/python/ops/_lsh_matmul_op.so
LSH_MATMUL_GRAD_TARGET_LIB = tensorflow_lsh_matmul/python/ops/lsh_matmul_grad.so
#LSH_MATMUL_GRAD_TARGET_LIB = lsh_matmul/python/ops/_lsh_matmul_grad_op.so

# zero_out op for CPU
#zero_out_op: $(ZERO_OUT_TARGET_LIB)

#$(ZERO_OUT_TARGET_LIB): $(ZERO_OUT_SRCS)
#	$(CXX) $(CFLAGS) -o $@ $^ ${LDFLAGS}

#zero_out_test: tensorflow_zero_out/python/ops/zero_out_ops_test.py tensorflow_zero_out/python/ops/zero_out_ops.py $(ZERO_OUT_TARGET_LIB)
#	$(PYTHON_BIN_PATH) tensorflow_zero_out/python/ops/zero_out_ops_test.py

zero_out_pip_pkg: $(ZERO_OUT_TARGET_LIB)
	./build_pip_pkg.sh make artifacts

# LSH MatMul op
lsh_matmul_op: $(LSH_MATMUL_TARGET_LIB)
lsh_matmul_grad_op: $(LSH_MATMUL_GRAD_TARGET_LIB)

$(LSH_MATMUL_TARGET_LIB): $(LSH_MATMUL_SRCS)
	$(CXX) $(CFLAGS) -o $@ $^ ${LDFLAGS}
    
$(LSH_MATMUL_GRAD_TARGET_LIB): $(LSH_MATMUL_GRAD_SRCS)
	$(CXX) $(CFLAGS) -o $@ $^ ${LDFLAGS}

lsh_matmul_test: tensorflow_lsh_matmul/python/ops/lsh_matmul_op_test.py tensorflow_lsh_matmul/python/ops/lsh_matmul_op.py $(LSH_MATMUL_TARGET_LIB) $(LSH_MATMUL_GRAD_TARGET_LIB)
	$(PYTHON_BIN_PATH) tensorflow_lsh_matmul/python/ops/lsh_matmul_op_test.py

lsh_matmul_pip_pkg: $(LSH_MATMUL_TARGET_LIB) $(LSH_MATMUL_GRAD_TARGET_LIB)
	./build_pip_pkg.sh make artifacts

clean:
	rm -f $(LSH_MATMUL_TARGET_LIB) $(LSH_MATMUL_GRAD_TARGET_LIB)
    #rm -f $(LSH_MATMUL_TARGET_LIB) $(LSH_MATMUL_GRAD_TARGET_LIB)
