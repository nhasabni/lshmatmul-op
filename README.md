# LSH-based implmentations of various TensorFlow layers

This repository contains implementation of various TensorFlow layers that can
benefit of using locality-sensitive hashing.

# Quick build and test default zero_out op

Build the pip package with make as:
```bash
   make zero_out_pip_pkg
```
Install the pip package as:
```bash
   pip3 install artifacts/*.whl
```
Test zero_out op as:
```bash
cd ..
python3 -c "import tensorflow as tf;import tensorflow_zero_out;print(tensorflow_zero_out.zero_out([[1,2], [3,4]]))"
```
And you should see the op zeroed out all input elements except the first one:
```bash
[[1 0]
 [0 0]]
```

# LSH-MatMul build and test (WIP)

Build the pip package with make as:
```bash
   make lsh_matmul_pip_pkg
```
Install the pip package as:
```bash
   pip3 install artifacts/*.whl
```
Test lsh_matmul op as:
```bash
TODO
```
And you should see the output as:
```bash
```
