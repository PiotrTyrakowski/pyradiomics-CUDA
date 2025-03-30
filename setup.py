#!/usr/bin/env python

from distutils import sysconfig
# Need distutils.ccompiler for gen_preprocess_options, gen_lib_options
from distutils import ccompiler
from distutils.errors import CompileError, LinkError # For error handling
import platform
import os
import sys
import subprocess
import shutil

import numpy
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext as _build_ext
import versioneer

# --- Helper Function to find NVCC ---
def find_nvcc():
    # Try finding nvcc in PATH
    nvcc_path = shutil.which('nvcc')
    if nvcc_path:
        print(f"Found nvcc at: {nvcc_path}")
        return nvcc_path

    # Try common install locations if not in PATH (optional)
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if cuda_home:
        print(f"CUDA_HOME/CUDA_PATH found at: {cuda_home}")
        # Check both bin and bin64 just in case
        for bindir in ['bin', 'bin64']:
            potential_path = os.path.join(cuda_home, bindir, 'nvcc')
            if os.path.exists(potential_path):
                print(f"Found nvcc via CUDA_HOME/PATH ({bindir}): {potential_path}")
                return potential_path
            potential_path += '.exe' # Try windows extension
            if os.path.exists(potential_path):
                 print(f"Found nvcc.exe via CUDA_HOME/PATH ({bindir}): {potential_path}")
                 return potential_path

    print("WARNING: nvcc command not found in PATH or via CUDA_HOME/CUDA_PATH.", file=sys.stderr)
    return None

# --- Custom build_ext Class ---
class CudaBuildExt(_build_ext):
    """ Custom build_ext command to handle .cu files by pre-compiling them. """
    def build_extensions(self):
        # Ensure nvcc is available if building CUDA extensions
        nvcc_path = None
        has_cuda_ext = any('.cu' in src for ext in self.extensions for src in ext.sources if isinstance(src, str))

        if has_cuda_ext:
            nvcc_path = find_nvcc()
            if nvcc_path is None:
                raise RuntimeError("nvcc compiler not found, cannot build CUDA extensions.")

        # Backup original compiler settings if needed (might not be necessary with this approach)
        # ...

        for ext in self.extensions:
            cuda_sources = []
            other_sources = []
            is_cuda_ext = False # Flag to track if this extension has CUDA files

            # Separate .cu files from other sources
            for source in ext.sources:
                if isinstance(source, str) and os.path.splitext(source)[1] == '.cu':
                    cuda_sources.append(source)
                    is_cuda_ext = True # Mark as CUDA extension
                else:
                    other_sources.append(source)

            # If there are .cu files, compile them to .o files first
            if is_cuda_ext: # Check the flag
                if not cuda_sources:
                     print(f"Warning: Extension {ext.name} marked as CUDA but no .cu files found?", file=sys.stderr)
                     # Decide how to handle this - maybe skip NVCC steps?
                     # For now, we'll let it proceed, but the NVCC logic below won't run.

                if cuda_sources:
                    print(f"Pre-compiling CUDA sources for extension {ext.name}...")
                    ext_build_dir = os.path.join(self.build_temp, ext.name) # Directory for object files
                    if not os.path.exists(ext_build_dir):
                        os.makedirs(ext_build_dir)

                    objects = []
                    nvcc_compile_args = [] # Store nvcc specific args separately

                    # Extract nvcc-specific args if present
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

                        # Add language standard if needed
                        # cmd.append('--std=c++14') # Example

                        # Add architecture flags and other nvcc args
                        cmd.extend(nvcc_compile_args) # Add flags collected earlier

                        print(f"Running nvcc command: {' '.join(cmd)}")
                        try:
                            # Capture stderr for better diagnostics
                            result = subprocess.run(cmd, check=True, text=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
                            if result.stderr:
                                print(f"nvcc stderr:\n{result.stderr}", file=sys.stderr)
                        except subprocess.CalledProcessError as e:
                            print(f"ERROR: nvcc compilation failed for {cuda_src}:\n{e.stderr}", file=sys.stderr)
                            raise CompileError(f"nvcc compilation failed for {cuda_src}")
                        except FileNotFoundError:
                            print(f"ERROR: nvcc command '{nvcc_path}' not found during compilation.", file=sys.stderr)
                            raise FileNotFoundError("nvcc not found during compilation")

                        objects.append(obj_path)

                # Replace .cu sources with other sources for the C/C++ compiler
                ext.sources = other_sources
                # Add the compiled .o files to be linked
                ext.extra_objects = objects # This is the list of pre-compiled objects

                # Ensure the C++ linker is used if CUDA was involved
                # (nvcc usually needs g++ or similar)
                ext.language = ext.language or 'c++'

                # *** THE FIX ***
                # Reset extra_compile_args for the C/C++ compiler.
                # It expects a list, not the dict we used for nvcc args.
                # If you had C/C++ specific flags, you'd filter them from the original
                # dict, but here we just need an empty list.
                ext.extra_compile_args = []


        # Now, run the original build_extensions method.
        # It will compile the remaining .c/.cpp files and link them with the .o files
        # specified in ext.extra_objects.
        print("Proceeding with standard build process for remaining sources and linking...")
        _build_ext.build_extensions(self)

# --- Main Setup Logic ---
if platform.architecture()[0].startswith('32'):
  raise Exception('PyRadiomics requires 64 bits python')

# Check if the user wants to build CUDA extensions
build_cuda = os.environ.get('BUILD_CUDA_EXTENSIONS', '0') == '1'

commands = versioneer.get_cmdclass()
commands['build_ext'] = CudaBuildExt

incDirs = [sysconfig.get_python_inc(), numpy.get_include()]

# --- CPU Extensions (Always included) ---
cpu_extensions = [
    Extension("radiomics._cmatrices",
              sources=["radiomics/src/_cmatrices.c", "radiomics/src/cmatrices.c"],
              include_dirs=incDirs),
    Extension("radiomics._cshape",
              sources=["radiomics/src/_cshape.c", "radiomics/src/cshape.c"],
              include_dirs=incDirs + [os.path.join('radiomics', 'src')]),
]

# --- CUDA Extensions (Conditionally included) ---
cuda_extensions = []
if build_cuda:
    print("BUILD_CUDA_EXTENSIONS=1 found. Attempting to include CUDA extensions.")
    cuda_src_dir = os.path.join('radiomics', 'src', 'cuda')
    cuda_inc_dirs = incDirs + [cuda_src_dir]

    # --- Determine CUDA Library Directory ---
    nvcc_path_for_libs = find_nvcc() # Reuse find_nvcc to get path
    cuda_library_dirs = [] # Default to empty list
    if nvcc_path_for_libs:
        # Try to infer lib64/lib path relative to nvcc's directory
        cuda_root = os.path.dirname(os.path.dirname(nvcc_path_for_libs)) # Go up two levels (bin -> root)
        potential_lib_paths = [
            os.path.join(cuda_root, 'lib64'),
            os.path.join(cuda_root, 'lib')
        ]
        found_lib_dir = None
        for lib_path in potential_lib_paths:
            if os.path.isdir(lib_path):
                found_lib_dir = lib_path
                print(f"Found CUDA library directory based on nvcc path: {found_lib_dir}")
                break

        if found_lib_dir:
            cuda_library_dirs.append(found_lib_dir)
        else:
            print(f"Warning: Could not automatically determine CUDA library directory relative to nvcc: {nvcc_path_for_libs}. Checking CUDA_HOME/PATH.", file=sys.stderr)
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
                         print(f"Using CUDA library directory from CUDA_HOME/PATH: {found_lib_dir_env}")
                         break
                 if found_lib_dir_env:
                     cuda_library_dirs.append(found_lib_dir_env)
                 else:
                     print(f"Warning: CUDA_HOME/PATH set ({cuda_home_env}), but could not find lib64 or lib subdir.", file=sys.stderr)

    if not cuda_library_dirs:
        print("Warning: CUDA library directory not found. Linking might fail if it's not in the system's default search paths.", file=sys.stderr)


    # --- Read NVCC_FLAGS from environment ---
    nvcc_flags_env = os.environ.get('NVCC_FLAGS', '')
    nvcc_extra_args = nvcc_flags_env.split() # Split flags by space
    if nvcc_extra_args:
        print(f"Using NVCC_FLAGS from environment: {nvcc_flags_env}")
    else:
        print("NVCC_FLAGS environment variable not set or empty. Using default nvcc options.")
        # Optionally add default flags here if needed, e.g.:
        # nvcc_extra_args = ['-arch=sm_60']

    cuda_shape_ext = Extension(
        "radiomics.cuda._cshape",
        sources=[
            os.path.join(cuda_src_dir, "_cshape.c"), # CPython interface (will be compiled by C compiler)
            os.path.join(cuda_src_dir, "cshape.cu")  # CUDA kernels (will be pre-compiled by nvcc)
        ],
        include_dirs=cuda_inc_dirs,
        libraries=['cudart'], # Link CUDA runtime library
        library_dirs=cuda_library_dirs,
        language='c++', # Ensure C++ linker is used
        extra_compile_args={'nvcc': nvcc_extra_args}
    )
    cuda_extensions.append(cuda_shape_ext)

else:
    print("BUILD_CUDA_EXTENSIONS not set or not '1'. Building CPU extensions only.")

# Combine extension lists
all_extensions = cpu_extensions + cuda_extensions

setup(
  name='pyradiomics',

  version=versioneer.get_version(),
  cmdclass=commands,

  packages=['radiomics', 'radiomics.scripts', 'radiomics.cuda'],
  ext_modules=all_extensions,
  zip_safe=False
)
