from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='qe_cpp',
      ext_modules=[cpp_extension.CppExtension('qe_cpp', ['qe.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
