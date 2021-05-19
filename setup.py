from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        'rfidam.baskets_mc',
        ['rfidam/baskets_mc.pyx'],
        include_dirs=[np.get_include()],
    ),
    Extension(
        'rfidam.cy_ext.simulation',
        ['rfidam/cy_ext/simulation.pyx'],
        language="c++",
        include_dirs=[np.get_include()],
    )
]

setup(
    name='rfidam',
    version='1.0',
    py_modules=['rfidam'],
    install_requires=[
        'click>=8.0.0',
        'numpy>=1.20.3',
        'scipy>=1.6.3',
    ],
    tests_requires=[
        'pytest',
    ],
    entry_points='''
        [console_scripts]
        rfidam=rfidam.main:main
    ''',
    ext_modules=cythonize(
        extensions,
        compiler_directives={'language_level': "2"},
        annotate=True
    ),
)
