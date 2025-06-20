from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='pointnet2',
    ext_modules=[
        CUDAExtension('pointnet2_cuda', [
            './fps/sample.cpp',
            './fps/sample_cuda.cu',
        ],
                      extra_compile_args={'cxx': ['-g'],
                                          'nvcc': ['-O2']})
    ],
    cmdclass={'build_ext': BuildExtension}
)