load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_custom_op_library")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "test_data",
    srcs = glob(["testdata/**"]),
)

cc_library(
    name = "icp_op_kernel",
    srcs = ["icp_op_kernel.cc"],
#    srcs = ["icp_op_kernel.cc eigen_3_3_4/*"],
    copts = [
        "-fexceptions",
        "-Wno-sign-compare",
        "-D_GLIBCXX_USE_CXX11_ABI=0",
        "-I/usr/include/pcl-1.7", 
        "-I/usr/include/eigen_3_3_4", 
        "-Iops/eigen_3_3_4", 
    ],
    linkopts = [" -lpcl_common -lpcl_features -lpcl_filters -lpcl_io -lpcl_io_ply -lpcl_kdtree -lpcl_keypoints -lpcl_octree -lpcl_outofcore -lpcl_people -lpcl_recognition -lpcl_registration -lpcl_sample_consensus -lpcl_search -lpcl_segmentation -lpcl_surface -lpcl_tracking -lpcl_visualization -lflann -L/usr/lib/x86_64-linux-gnu" ],
    deps = [
    #    "@com_github_pointcloudlibrary_pcl//:common",
    #    "@com_github_pointcloudlibrary_pcl//:registration",
        "@com_google_protobuf//:protobuf",
        "@org_tensorflow//tensorflow/core:framework_headers_lib",
    ],
)

tf_custom_op_library(
    name = "icp_op.so",
    linkopts = ["-llz4"],
    deps = [
        ":icp_op_kernel",
    ],
)

py_library(
    name = "icp_op",
    srcs = ["icp_op.py"],
    data = [
        ":icp_op.so",
    ],
    srcs_version = "PY2AND3",
    deps = [
    ],
)

py_library(
    name = "icp_util",
    srcs = ["icp_util.py"],
    data = [":test_data"],
    srcs_version = "PY2AND3",
    deps = [
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "icp_grad",
    srcs = ["icp_grad.py"],
    data = [
        ":icp_op.so",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "@org_tensorflow//tensorflow:tensorflow_py",
        ":icp_op",
    ],
)

cc_binary(
    name = "pcl_demo",
    srcs = ["pcl_demo.cc"],
    copts = [
        "-I/usr/include/pcl-1.7", 
        "-I/usr/include/eigen3", 
    ],
    linkopts = [" -lpcl_common -lpcl_features -lpcl_filters -lpcl_io -lpcl_io_ply -lpcl_kdtree -lpcl_keypoints -lpcl_octree -lpcl_outofcore -lpcl_people -lpcl_recognition -lpcl_registration -lpcl_sample_consensus -lpcl_search -lpcl_segmentation -lpcl_surface -lpcl_tracking -lpcl_visualization -lflann -L/usr/lib/x86_64-linux-gnu" ],
    deps = [
#        "@com_github_pointcloudlibrary_pcl//:common",
#        "@com_github_pointcloudlibrary_pcl//:registration",
    ],
)

py_binary(
    name = "icp_train_demo",
    srcs = ["icp_train_demo.py"],
    data = [
        ":icp_op.so",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "@org_tensorflow//tensorflow:tensorflow_py",
        ":icp_op",
        ":icp_grad",
        ":icp_util",
    ],
)

py_test(
    name = "icp_test",
    size = "small",
    srcs = ["icp_test.py"],
    data = [
        ":icp_op.so",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "@org_tensorflow//tensorflow:tensorflow_py",
        ":icp_op",
        ":icp_util",
    ],
)

py_test(
    name = "icp_grad_test",
    size = "small",
    srcs = ["icp_grad_test.py"],
    data = [
        ":icp_op.so",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "@org_tensorflow//tensorflow:tensorflow_py",
        ":icp_op",
        ":icp_grad",
        ":icp_test",
    ],
)
