licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "windows",
    constraint_values = ["@bazel_tools//platforms:windows"],
)

cc_binary(
    name = 'python/ops/_lsh_matmul_op.so',
    srcs = [
        "cc/kernels/lsh_matmul_kernels.cc",
        "cc/ops/lsh_matmul_op.cc",
    ],
    linkshared = 1,
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
    features = select({
        ":windows": ["windows_export_all_symbols"],
        "//conditions:default": [],
    }),
    copts = select({
        ":windows": ["/DEIGEN_STRONG_INLINE=inline", "-DTENSORFLOW_MONOLITHIC_BUILD", "/DPLATFORM_WINDOWS", "/DEIGEN_HAS_C99_MATH", "/DTENSORFLOW_USE_EIGEN_THREADPOOL", "/DEIGEN_AVOID_STL_ARRAY", "/Iexternal/gemmlowp", "/wd4018", "/wd4577", "/DNOGDI", "/UTF_COMPILE_LIBRARY"],
        "//conditions:default": ["-pthread", "-std=c++11", "-D_GLIBCXX_USE_CXX11_ABI=0"],
    }),
)

py_library(
    name = "lsh_matmul_op_py",
    srcs = ([
        "python/ops/lsh_matmul_op.py",
    ]),
    data = [
        ":python/ops/_lsh_matmul_op.so"
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "lsh_matmul_op_py_test",
    srcs = [
        "python/ops/lsh_matmul_op_test.py"
    ],
    main = "python/ops/lsh_matmul_op_test.py",
    deps = [
        ":lsh_matmul_op_py",
    ],
    srcs_version = "PY2AND3",
)

py_library(
    name = "lsh_matmul_py",
    srcs = ([
        "__init__.py",
        "python/__init__.py",
        "python/ops/__init__.py",
    ]),
    deps = [
        ":lsh_matmul_op_py"
    ],
    srcs_version = "PY2AND3",
)
