import os
import subprocess
from packaging.version import parse, Version

from setuptools import setup

import torch
from torch.utils.cpp_extension import (
    BuildExtension,
    CppExtension,
    CUDAExtension,
    CUDA_HOME,
    load,
)


def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_version = parse(output[release_idx].split(",")[0])

    return raw_output, bare_metal_version


def check_cuda_torch_binary_vs_bare_metal(cuda_dir):
    raw_output, bare_metal_version = get_cuda_bare_metal_version(cuda_dir)
    torch_binary_version = parse(torch.version.cuda)

    print("\nCompiling cuda extensions with")
    print(raw_output + "from " + cuda_dir + "/bin\n")

    if (bare_metal_version != torch_binary_version):
        raise RuntimeError(
            "Cuda extensions are being compiled with a version of Cuda that does "
            "not match the version used to compile Pytorch binaries.  "
            "Pytorch binaries were compiled with Cuda {}.\n".format(torch.version.cuda)
            + "In some cases, a minor-version mismatch will not cause later errors:  "
            "https://github.com/NVIDIA/apex/pull/323#discussion_r287021798.  "
            "You can try commenting out this check (at your own risk)."
        )


def raise_if_cuda_home_none() -> None:
    if CUDA_HOME is not None:
        return
    raise RuntimeError(
        "nvcc was not found.  Are you sure your environment has nvcc available?  "
        "If you're installing within a container from https://hub.docker.com/r/pytorch/pytorch, "
        "only images whose names contain 'devel' will provide nvcc."
    )


def check_cupti_directory(cuda_dir):
    cupti_dir = os.path.join(cuda_dir, 'extras', 'CUPTI')
    if not os.path.isdir(cupti_dir):
        raise RuntimeError(
            f"CUPTI not found in {cupti_dir}. Please ensure that CUPTI is installed "
            "as part of your CUDA installation. CUPTI is required for profiling."
        )
    return cupti_dir


# Ensure CUDA_HOME is set and the directories are valid
raise_if_cuda_home_none()

# Check if CUDA version matches PyTorch binaries
check_cuda_torch_binary_vs_bare_metal(CUDA_HOME)

# Check if CUPTI directory exists within CUDA_HOME
cupti_dir = check_cupti_directory(CUDA_HOME)

# Setup extension
setup(
    name='vtrain_profiler',
    ext_modules=[
        CppExtension(
            name='vtrain_profiler',
            sources=['cupti.cpp'],
            include_dirs=[
                os.path.join(cupti_dir, 'include'),
                os.path.join(CUDA_HOME, 'targets', 'x86_64-linux', 'include'),
            ],
            library_dirs=[
                os.path.join(CUDA_HOME, 'lib64'),
                os.path.join(cupti_dir, 'lib64'),
            ],
            libraries=['cupti']
            ),
        ],
    cmdclass={
        'build_ext': BuildExtension,
    }
)
