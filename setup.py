import os
import re
import sysconfig
import setuptools
import glob
from setuptools import find_packages
from setuptools.command.build_py import build_py
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME, include_paths

current_dir = os.path.dirname(os.path.realpath(__file__))

def get_all_files(directory):
    all_files = glob.glob(os.path.join(directory, "**", "*"), recursive=True)
    files = [f[len(current_dir)+1:] for f in all_files if os.path.isfile(f) and (f.endswith(".cc") or f.endswith(".cu"))]
    return files

sources = get_all_files(os.path.join(current_dir, 'csrc'))

build_include_dirs = [
    f'{current_dir}/include',
    f'{current_dir}/csrc',
    f'{current_dir}/thirdparty/json/include',
    f'{current_dir}/thirdparty/asio/include',
    f'{current_dir}/thirdparty/spdlog/include',
    *include_paths(),
]

nvtx_wheel_include = os.path.join(
    sysconfig.get_paths()["purelib"], "nvidia", "nvtx", "include")
if os.path.isdir(nvtx_wheel_include):
    build_include_dirs.append(nvtx_wheel_include)

build_libraries = []

build_library_dirs = []

rdma_enabled = os.path.exists('/usr/include/infiniband/verbs.h') or os.path.exists('/usr/include/rdma/rdma_cma.h')

if CUDA_HOME:
    build_include_dirs.append(f'{CUDA_HOME}/include')
    build_libraries.extend(['cuda', 'cudart', 'nvrtc', 'nvToolsExt'])
    build_library_dirs.extend([
        f'{CUDA_HOME}/lib',
        f'{CUDA_HOME}/lib/stubs',
        f'{CUDA_HOME}/lib64',
        f'{CUDA_HOME}/lib64/stubs',
        f'{CUDA_HOME}/targets/x86_64-linux/lib/stubs/',
    ])
    if any(os.path.exists(os.path.join(directory, 'libnvtx3interop.so'))
           for directory in build_library_dirs):
        build_libraries.append('nvtx3interop')

cxx_flags = ['-std=c++17',
             '-fPIC',
             '-fvisibility=hidden',
             '-DASIO_STANDALONE',
             '-DASIO_HEADER_ONLY',
             '-DFMT_HEADER_ONLY',
             '-g',
             '-mavx2',
             '-mfma',
             '-fopenmp',
            ]

if rdma_enabled:
    cxx_flags.append('-DPCCL_RDMA_ENABLED')
    build_libraries.append('ibverbs')

cuda_arch = os.environ.get('PCCL_CUDA_ARCH', '90a')
if not re.fullmatch(r'\d+a?', cuda_arch):
    raise ValueError('PCCL_CUDA_ARCH must be a CUDA capability such as 86 or 90a')

cuda_flags = ['-std=c++17',
              '-O3',
              f'-gencode=arch=compute_{cuda_arch},code=sm_{cuda_arch}',
              '-lineinfo',
             ]

if os.environ.get('PCCL_ALLOW_UNSUPPORTED_COMPILER') == '1':
    cuda_flags.append('-allow-unsupported-compiler')

if __name__ == '__main__':

    setuptools.setup(
        name='pccl',
        version='0.3.0',
        packages=find_packages('.'),
        ext_modules=[
            CUDAExtension(name='pccl.engine_c',
                         sources=sources,
                         include_dirs=build_include_dirs,
                         libraries=build_libraries,
                         library_dirs=build_library_dirs,
                         extra_compile_args={'cxx': cxx_flags,
                                             'nvcc': cuda_flags,
                                            },
            )
        ],
        zip_safe=False,
        cmdclass={
            'build_ext': BuildExtension,
            'build_py': build_py,
        },
    )
