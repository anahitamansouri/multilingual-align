
from distutils.core import setup
from Cython.Build import cythonize

setup(
  name = 'Featurized HMM Parallel with E step MP',
  ext_modules = cythonize("featurized_hmm_mp_e_step_parallel_theta_efficient.pyx"),
)
