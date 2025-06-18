#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include "sample_cuda.h"


int furthest_point_sampling_wrapper(int b, int N, int K, at::Tensor pts, at::Tensor temp_tensor, at::Tensor idx_tensor) {
	/*
	Inputs:
	b: Batch的值
	n: 原始点云中点的数量
	m: 要选取点的数量
	points_tensor: 原始点云，大小为n*3
	temp_tensor: 中间变量，大小为n*3
	idx_tensor: 这个是返回值，储存选取的idx, 大小为m*3

	points_tensor, temp_tensor, idx_tensor都是在cuda上面的tensor
	*/
    const float *points = pts.data<float>();
    float *temp = temp_tensor.data<float>();
    int *idx = idx_tensor.data<int>();
    // 使用ATen的API获取当前CUDA流
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
	// 调用sampling_gpu.cu中的函数
    furthest_point_sampling_kernel_launcher(b, N, K, points, temp, idx, stream);
    return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
    "furthest_point_sampling_wrapper",
    &furthest_point_sampling_wrapper,
    "furthest_point_sampling_wrapper"
  );
}