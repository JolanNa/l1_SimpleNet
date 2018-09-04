# Description:
#   Contains the operations and nets for building TensorFlow-Slim models.

package(default_visibility = ["//simple_net:internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])





py_library(
    name = "ops",
    srcs = ["ops.py"],
    deps = [
        ":activation_functions",
    ],
)

py_library(
    name = "slim",
    srcs = ["slim.py"],
    deps = [
        ":simple_net_model",
        ":ops",
        ":activation_functions",
    ],
)

py_library(
    name = "simple_net_model",
    srcs = ["simple_net_model.py"],
    deps = [
        ":ops",
    ],
)