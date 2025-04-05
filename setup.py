#!/usr/bin/env python

import os
import platform
import shutil
import subprocess
import sys
from distutils import sysconfig
from distutils.errors import CompileError

import numpy
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext as _build_ext

import versioneer


# -------------------------------
# Cuda setuptools extension
# -------------------------------

def find_nvcc():
  nvcc_path = shutil.which('nvcc')
  if nvcc_path:
    return nvcc_path

  cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
  if cuda_home:

    for bindir in ['bin', 'bin64']:
      potential_path = os.path.join(cuda_home, bindir, 'nvcc')

      if os.path.exists(potential_path):
        return potential_path

      potential_path += '.exe'
      if os.path.exists(potential_path):
        return potential_path

  return None


# --- Custom build_ext Class ---
class CudaBuildExt(_build_ext):
  """ Custom build_ext command to handle .cu files by pre-compiling them. """

  def build_extensions(self):
    nvcc_path = None
    has_cuda_ext = any(
      '.cu' in src for ext in self.extensions for src in ext.sources if isinstance(src, str))

    if has_cuda_ext:
      nvcc_path = find_nvcc()

      if nvcc_path is None:
        raise RuntimeError("nvcc compiler not found, cannot build CUDA extensions.")

    for ext in self.extensions:
      cuda_sources = []
      other_sources = []

      # Separate .cu files from other sources
      for source in ext.sources:
        if isinstance(source, str) and os.path.splitext(source)[1] == '.cu':
          cuda_sources.append(source)
        else:
          other_sources.append(source)

      if cuda_sources:
        ext_build_dir = os.path.join(self.build_temp, ext.name)
        if not os.path.exists(ext_build_dir):
          os.makedirs(ext_build_dir)

        objects = []
        nvcc_compile_args = []

        if isinstance(ext.extra_compile_args, dict) and 'nvcc' in ext.extra_compile_args:
          nvcc_compile_args = ext.extra_compile_args['nvcc']

        for cuda_src in cuda_sources:
          base_name = os.path.basename(cuda_src)
          obj_name = os.path.splitext(base_name)[0] + '.o'
          obj_path = os.path.join(ext_build_dir, obj_name)

          # Build the nvcc command
          cmd = [nvcc_path, '-c', cuda_src, '-o', obj_path]

          # Add include directories from the extension and the build command
          all_includes = ext.include_dirs + self.include_dirs
          for include_dir in all_includes:
            cmd.extend(['-I', include_dir])

          # Add Position Independent Code flag needed for shared libraries
          cmd.append('-Xcompiler=-fPIC')

          # Add flags collected earlier
          cmd.extend(nvcc_compile_args)

          try:
            print(f"Compiling {cuda_src} with nvcc: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, text=True, stderr=subprocess.PIPE,
                                    stdout=subprocess.PIPE)

          except subprocess.CalledProcessError as e:
            print(f"ERROR: nvcc compilation failed for {cuda_src}:\n{e.stderr}", file=sys.stderr)
            raise CompileError(f"nvcc compilation failed for {cuda_src}")

          except FileNotFoundError:
            print(f"ERROR: nvcc command '{nvcc_path}' not found during compilation.",
                  file=sys.stderr)
            raise FileNotFoundError("nvcc not found during compilation")

          objects.append(obj_path)

        # Replace .cu sources with other sources for the C/C++ compiler
        ext.sources = other_sources

        # Add the compiled .o files to be linked
        ext.extra_objects = objects

        ext.extra_compile_args = ext.extra_compile_args.get('C', []) if \
          isinstance(ext.extra_compile_args, dict) else ext.extra_compile_args

    _build_ext.build_extensions(self)


def attach_cuda_if_possible(extension, cuda_sources):
  if os.environ.get('DISABLE_CUDA_EXTENSIONS', '0') == '1':
    print("DISABLE_CUDA_EXTENSIONS=1: skipping CUDA extension build")
    return extension

  nvcc = find_nvcc()
  if not nvcc:
    print("CUDA not found. Skipping CUDA extension build.")
    return extension

  # --- Determine CUDA Library Directory ---
  nvcc_path_for_libs = nvcc
  cuda_library_dirs = []

  if nvcc_path_for_libs:
    cuda_root = os.path.dirname(
      os.path.dirname(nvcc_path_for_libs))

    potential_lib_paths = [
      os.path.join(cuda_root, 'lib64'),
      os.path.join(cuda_root, 'lib')
    ]

    found_lib_dir = None
    for lib_path in potential_lib_paths:
      if os.path.isdir(lib_path):
        found_lib_dir = lib_path
        break

    if found_lib_dir:
      cuda_library_dirs.append(found_lib_dir)
    else:
      print(
        f"Warning: Could not automatically determine CUDA library directory relative to nvcc: {nvcc_path_for_libs}. Checking CUDA_HOME/PATH.",
        file=sys.stderr)

      # Fallback to CUDA_HOME/PATH environment variables if relative path failed
      cuda_home_env = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')

      if cuda_home_env:
        potential_lib_paths_env = [
          os.path.join(cuda_home_env, 'lib64'),
          os.path.join(cuda_home_env, 'lib')
        ]

        found_lib_dir_env = None
        for lib_path_env in potential_lib_paths_env:
          if os.path.isdir(lib_path_env):
            found_lib_dir_env = lib_path_env
            break

        if found_lib_dir_env:
          cuda_library_dirs.append(found_lib_dir_env)
        else:
          print(
            f"Warning: CUDA_HOME/PATH set ({cuda_home_env}), but could not find lib64 or lib subdir.",
            file=sys.stderr)

  if not cuda_library_dirs:
    print(
      "Warning: CUDA library directory not found. Linking might fail if it's not in the system's default search paths.",
      file=sys.stderr)

  # --- Read NVCC_FLAGS from environment ---
  nvcc_flags_env = os.environ.get('NVCC_FLAGS', '')
  nvcc_extra_args = nvcc_flags_env.split()

  if not nvcc_extra_args:
    nvcc_extra_args = ['-O3']

  cuda_include_dir = os.path.join('radiomics', 'src', 'cuda')
  extension.sources += cuda_sources
  extension.include_dirs += [cuda_include_dir]
  extension.libraries += ['cudart']
  extension.library_dirs += cuda_library_dirs
  extension.extra_compile_args = {
    'nvcc': nvcc_extra_args,
    'C': extension.extra_compile_args
  }
  extension.define_macros += [('CUDA_EXTENSIONS_ENABLED', '1')]

  return extension

# ------------------------------
# Actual configuration
# ------------------------------

if platform.architecture()[0].startswith('32'):
  raise Exception('PyRadiomics requires 64 bits python')

commands = versioneer.get_cmdclass()
commands['build_ext'] = CudaBuildExt
incDirs = [sysconfig.get_python_inc(), numpy.get_include()]

ext = [
  Extension("radiomics._cmatrices", ["radiomics/src/_cmatrices.c", "radiomics/src/cmatrices.c"],
                 include_dirs=incDirs),
  attach_cuda_if_possible(
    extension=Extension("radiomics._cshape", ["radiomics/src/_cshape.c", "radiomics/src/cshape.c"],
                        include_dirs=incDirs),
    cuda_sources=["radiomics/src/cuda/cshape.cu"] + [
      os.path.join("radiomics/src/cuda/implementations", f)
      for f in os.listdir("radiomics/src/cuda/implementations")
    ]
  )
]

setup(
  name='pyradiomics',

  version=versioneer.get_version(),
  cmdclass=commands,

  packages=['radiomics', 'radiomics.scripts'],
  ext_modules=ext,
  zip_safe=False
)
