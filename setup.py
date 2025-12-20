import os
import setuptools
import shutil
import subprocess
import glob
import subprocess
from setuptools import find_packages
from setuptools.command.build_py import build_py
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDA_HOME

current_dir = os.path.dirname(os.path.realpath(__file__))

def get_all_files(directory):
    all_files = glob.glob(os.path.join(directory, "**", "*"), recursive=True)
    files = [f[len(current_dir)+1:] for f in all_files if os.path.isfile(f)]
    return files

sources = get_all_files(os.path.join(current_dir, 'csrc'))

def find_rocm():
    rocm_path = os.environ.get('ROCM_PATH', '/opt/rocm')
    if os.path.isdir(rocm_path):
        return rocm_path
    return None

rocm_path = find_rocm()
has_rocm = rocm_path is not None

build_include_dirs = [
    f'{CUDA_HOME}/include',
    f'{current_dir}/include',
    f'{current_dir}/thirdparty/cutlass/include',
    f'{current_dir}/thirdparty/composable_kernel/include',
    f'{current_dir}/thirdparty/json/include',
    f'{current_dir}/thirdparty/spdlog/include',
    f'{current_dir}/thirdparty/asio/asio/include',
]

build_libraries = ['cuda', 'cudart', 'nvrtc']
build_library_dirs = [
    f'{CUDA_HOME}/lib64',
    f'{CUDA_HOME}/lib64/stubs',
    f'{CUDA_HOME}/targets/x86_64-linux/lib/stubs/',
]

if has_rocm:
    build_include_dirs.extend([
        f'{rocm_path}/include',
        f'{rocm_path}/include/hip',
        f'{rocm_path}/include/rocblas',
        f'{rocm_path}/include/rocrand'
    ])
    build_libraries.extend(['hip', 'rocblas', 'rocrand', 'hiprtc'])
    build_library_dirs.extend([
        f'{rocm_path}/lib',
        f'{rocm_path}/lib64'
    ])

cxx_flags = ['-std=c++20',
             '-fPIC',
             '-fvisibility=hidden',
             '-DASIO_STANDALONE',
             '-DASIO_HEADER_ONLY',
            ]

cuda_flags = ['-std=c++20',
              '-O3',
             ]

debug = True
if debug:
    cxx_flags.extend(['-DPCCL_DEBUG'])
    cuda_flags.extend(['-DPCCL_DEBUG'])

if has_rocm:
    cxx_flags.extend(['-DPCCL_ROCM_SUPPORT=1'])
    cuda_flags.extend(['-DPCCL_ROCM_SUPPORT=1'])

if __name__ == '__main__':
    try:
        cmd = ['git', 'rev-parse', '--short', 'HEAD']
        revision = '+' + subprocess.check_output(cmd).decode('ascii').rstrip()
    except:
        revision = ''

    extensions = [
        CppExtension(name='pccl.engine_c',
                     sources=sources,
                     include_dirs=build_include_dirs,
                     libraries=build_libraries,
                     library_dirs=build_library_dirs,
                     extra_compile_args={'cxx': cxx_flags,
                                         'cuda': cuda_flags},
        )
    ]

    cpu_sources = [
        'csrc/plugins/cpu_executor/device.cc',
        'csrc/plugins/cpu_executor/executor.cc',
        'csrc/plugins/cpu_executor/python_bindings.cc'
    ]

    cpu_include_dirs = build_include_dirs + [
        f'{current_dir}/include/plugins/cpu_executor'
    ]

    extensions.append(
        CppExtension(name='pccl_native_cpu',
                     sources=cpu_sources,
                     include_dirs=cpu_include_dirs,
                     libraries=build_libraries,
                     library_dirs=build_library_dirs,
                     extra_compile_args={'cxx': cxx_flags,
                                         'nvcc': []},
        )
    )

    if CUDA_HOME:
        cuda_sources = [
            'csrc/plugins/cuda_executor/device.cc',
            'csrc/plugins/cuda_executor/executor.cc',
            'csrc/plugins/cuda_executor/python_bindings.cc'
        ]

        cuda_include_dirs = build_include_dirs + [
            f'{current_dir}/include/plugins/cuda_executor'
        ]

        extensions.append(
            CppExtension(name='pccl_native_cuda',
                         sources=cuda_sources,
                         include_dirs=cuda_include_dirs,
                         libraries=build_libraries,
                         library_dirs=build_library_dirs,
                         extra_compile_args={'cxx': cxx_flags,
                                             'nvcc': []},
            )
        )

    try:
        import subprocess
        result = subprocess.run(['pkg-config', '--exists', 'libibverbs'],
                              capture_output=True)
        if result.returncode == 0:
            rdma_sources = [
                'csrc/plugins/rdma_executor/device.cc',
                'csrc/plugins/rdma_executor/executor.cc',
                'csrc/plugins/rdma_executor/utils/rdma_utils.cc',
                'csrc/plugins/rdma_executor/python_bindings.cc'
            ]

            rdma_include_dirs = build_include_dirs + [
                f'{current_dir}/include/plugins/rdma_executor',
                f'{current_dir}/csrc/plugins/rdma_executor/utils'
            ]

            rdma_libs_result = subprocess.run(['pkg-config', '--libs', 'libibverbs', 'librdmacm'],
                                            capture_output=True, text=True)
            rdma_libs = rdma_libs_result.stdout.strip().split() if rdma_libs_result.returncode == 0 else ['ibverbs', 'rdmacm']

            rdma_lib_dirs_result = subprocess.run(['pkg-config', '--libs-only-L', 'libibverbs', 'librdmacm'],
                                                capture_output=True, text=True)
            rdma_lib_dirs = []
            if rdma_lib_dirs_result.returncode == 0:
                for lib_dir in rdma_lib_dirs_result.stdout.strip().split():
                    if lib_dir.startswith('-L'):
                        rdma_lib_dirs.append(lib_dir[2:])

            extensions.append(
                CppExtension(name='pccl_native_rdma',
                             sources=rdma_sources,
                             include_dirs=rdma_include_dirs,
                             libraries=rdma_libs,
                             library_dirs=rdma_lib_dirs,
                             extra_compile_args={'cxx': cxx_flags,
                                                 'nvcc': []},
                )
            )
    except:
        pass

    if has_rocm:
        rocm_sources = [
            'csrc/plugins/rocm_executor/device.cc',
            'csrc/plugins/rocm_executor/executor.cc',
            'csrc/plugins/rocm_executor/utils/rocm_utils.cc',
            'csrc/plugins/rocm_executor/python_bindings.cc'
        ]

        rocm_include_dirs = build_include_dirs + [
            f'{current_dir}/include/plugins/rocm_executor',
            f'{current_dir}/csrc/plugins/rocm_executor/utils'
        ]

        extensions.append(
            CppExtension(name='pccl_native_rocm',
                         sources=rocm_sources,
                         include_dirs=rocm_include_dirs,
                         libraries=build_libraries,
                         library_dirs=build_library_dirs,
                         extra_compile_args={'cxx': cxx_flags,
                                             'nvcc': []},
            )
        )

    setuptools.setup(
        name='pccl',
        version='0.1.0' + revision,
        packages=find_packages('.'),
        package_data={
            'pccl': [
                'include/pccl/**/*',
                'include/cute/**/*',
                'include/cutlass/**/*',
                'include/ck/**/*',
                'include/ck_tile/**/*',
                'lib/*'
            ]
        },
        ext_modules=extensions,
        zip_safe=False,
        cmdclass={
            'build_ext': BuildExtension,
            'build_py': build_py,
        },
    )
