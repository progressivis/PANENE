from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy as np
import os
from pathlib import Path


HERE = Path(os.path.dirname(os.path.abspath(__file__)))
THERE = HERE / "python"
extension = [Extension(
    "pynene",
    [str(THERE / "pynene.pyx")],
    include_dirs= [str(THERE / "include"),
                   str(THERE / "cpp_src"),
                   str(HERE / "include"),
                   str(HERE / "examples" / "tsne" / "lib"),
                   np.get_include(),],
    extra_compile_args=["-std=c++17", "-w", "-g", "-O0"],
    language="c++",
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"), ("NDEBUG", "1")],
    # undef_macros = [ "NDEBUG" ]
)]

setup(name='pynene',
      version='0.1.0',
      description='Progressive Library for Approximate Nearest Neighbors',
      author='Jaemin Jo',
      author_email='jmjo@hcil.snu.ac.kr',
      license='BSD',
      url='https://github.com/e-/PANENE',
      package_dir={ '': str(THERE) },
#      packages=['pynene'],
      classifiers=[
          "Development Status :: 2 - PRe-Alpha",
          "Topic :: Scientific/Engineering :: Visualization",
          "Topic :: Scientific/Engineering :: Information Analysis",
      ],
    platforms='any',
    install_requires = [
        "numpy>=1.11.3",
        "scipy>=0.18.1",
        "cython>=0.25.1",
    ],

    ext_modules = cythonize(extension),
    zip_safe = False
)
