from distutils.core import setup, Extension

module = Extension('loopy', sources=['src/loopy_belief_propagation/loopy.cpp'], extra_compile_args=['-std=c++11'])

setup (name = 'loopy',
       version = '1.0',
       ext_modules = [module])