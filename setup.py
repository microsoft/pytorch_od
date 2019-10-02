#!/usr/bin/python

from __future__ import print_function
import os
import sys
import os.path as op
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
try:
    from torch.utils.cpp_extension import CUDAExtension, CUDA_HOME
    assert CUDA_HOME, "No CUDA found"
except (ImportError, OSError, AssertionError) as e:
    CUDAExtension = None
    print("No CUDA was detected, building without CUDA error: {}".format(e))

# change directory to this module path
try:
    this_file = __file__
except NameError:
    this_file = sys.argv[0]
this_file = os.path.abspath(this_file)
if op.dirname(this_file):
    os.chdir(op.dirname(this_file))
script_dir = os.getcwd()

include_dirs = [op.abspath('./mictorch/common/')]


def readme(fname):
    """Read text out of a file in the same directory as setup.py.
    """
    return open(op.join(script_dir, fname)).read()


if CUDAExtension is None:
    cuda_extensions = []
else:
    cuda_extensions = [
        CUDAExtension('smt_cuda', [
            'mictorch/smt/smt_cuda.cpp',
            'mictorch/smt/smt_cuda_kernel.cu',
        ], include_dirs=include_dirs),
        CUDAExtension('smtpred_cuda', [
            'mictorch/smtpred/smtpred_cuda.cpp',
            'mictorch/smtpred/smtpred_cuda_kernel.cu',
        ], include_dirs=include_dirs),
        CUDAExtension('nmsfilt_cuda', [
            'mictorch/nmsfilt/nmsfilt_cuda.cpp',
            'mictorch/nmsfilt/nmsfilt_cuda_kernel.cu',
        ], include_dirs=include_dirs),
    ]

setup(
    name="mictorch",
    version="0.0.1",
    author="ehazar",
    author_email="ehazar@microsoft.com",
    url='',
    description="Microsoft PyTorch object detection modules",
    long_description=readme('README.md'),
    packages=find_packages(),
    ext_modules=[
        CppExtension('smt_cpu', [
            'mictorch/smt/smt_cpu.cpp',
        ], include_dirs=include_dirs),
        CppExtension('nmsfilt_cpu', [
            'mictorch/nmsfilt/nmsfilt_cpu.cpp',
        ], include_dirs=include_dirs),
        CppExtension('smtpred_cpu', [
            'mictorch/smtpred/smtpred_cpu.cpp',
        ], include_dirs=include_dirs),
        CppExtension('nms_cpu', [
            'mictorch/nms/nms_cpu.cpp',
        ], include_dirs=include_dirs),
    ]  + cuda_extensions,
    cmdclass={
        'build_ext': BuildExtension
    },
    zip_safe=False,
    license="MIT",
    classifiers=[
        'Intended Audience :: Developers',
        "Programming Language :: Python",
        'Topic :: Software Development',
    ]
)
