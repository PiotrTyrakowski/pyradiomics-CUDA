# For convenience, import collection and numpy
# into the "pyradiomics" namespace

import collections  # noqa: F401
import inspect
import logging
import os
import pkgutil
import sys
import tempfile
import subprocess

import numpy  # noqa: F401
from six.moves import urllib

from . import imageoperations
from ._version import get_versions 




def deprecated(func):
  """
  Decorator function to mark functions as deprecated. This is used to ensure deprecated feature functions are not
  added to the enabled features list when enabling 'all' features.
  """
  func._is_deprecated = True
  return func


def setVerbosity(level):
  """
  Change the amount of information PyRadiomics should print out during extraction. The lower the level, the more
  information is printed to the output (stderr).

  Using the ``level`` (Python defined logging levels) argument, the following levels are possible:

  - 60: Quiet mode, no messages are printed to the stderr
  - 50: Only log messages of level "CRITICAL" are printed
  - 40: Log messages of level "ERROR" and up are printed
  - 30: Log messages of level "WARNING" and up are printed
  - 20: Log messages of level "INFO" and up are printed
  - 10: Log messages of level "DEBUG" and up are printed (i.e. all log messages)

  By default, the radiomics logger is set to level "INFO" and the stderr handler to level "WARNING". Therefore a log
  storing the extraction log messages from level "INFO" and up can be easily set up by adding an appropriate handler to
  the radiomics logger, while the output to stderr will still only contain warnings and errors.

  .. note::

    This function assumes the handler added to the radiomics logger at initialization of the toolbox is not removed from
    the logger handlers and therefore remains the first handler.

  .. note::

    This does not affect the level of the logger itself (e.g. if verbosity level = 3, log messages with DEBUG level can
    still be stored in a log file if an appropriate handler is added to the logger and the logging level of the logger
    has been set to the correct level. *Exception: In case the verbosity is set to DEBUG, the level of the logger is
    also lowered to DEBUG. If the verbosity level is then raised again, the logger level will remain DEBUG.*
  """
  global logger, handler
  if level < 10:  # Lowest level: DEBUG
    level = 10
  if level > 60:  # Highest level = 50 (CRITICAL), level 60 results in a 'quiet' mode
    level = 60

  handler.setLevel(level)
  if handler.level < logger.level:  # reduce level of logger if necessary
    logger.setLevel(level)


def getFeatureClasses():
  """
  Iterates over all modules of the radiomics package using pkgutil and subsequently imports those modules.

  Return a dictionary of all modules containing featureClasses, with modulename as key, abstract
  class object of the featureClass as value. Assumes only one featureClass per module

  This is achieved by inspect.getmembers. Modules are added if it contains a member that is a class,
  with name starting with 'Radiomics' and is inherited from :py:class:`radiomics.base.RadiomicsFeaturesBase`.

  This iteration only runs once (at initialization of toolbox), subsequent calls return the dictionary created by the
  first call.
  """
  global _featureClasses
  if _featureClasses is None:  # On first call, enumerate possible feature classes and import PyRadiomics modules
    _featureClasses = {}
    for _, mod, _ in pkgutil.iter_modules([os.path.dirname(__file__)]):
      if str(mod).startswith('_'):  # Skip loading of 'private' classes, these don't contain feature classes
        continue
      __import__('radiomics.' + mod)
      module = sys.modules['radiomics.' + mod]
      attributes = inspect.getmembers(module, inspect.isclass)
      for a in attributes:
        if a[0].startswith('Radiomics'):
          for parentClass in inspect.getmro(a[1])[1:]:  # only include classes that inherit from RadiomicsFeaturesBase
            if parentClass.__name__ == 'RadiomicsFeaturesBase':
              _featureClasses[mod] = a[1]
              break

  return _featureClasses


def getImageTypes():
  """
  Returns a list of possible image types (i.e. the possible filters and the "Original", unfiltered image type). This
  function finds the image types dynamically by matching the signature ("get<imageType>Image") against functions defined
  in :ref:`imageoperations <radiomics-imageoperations-label>`. Returns a list containing available image type names
  (<imageType> part of the corresponding function name).

  This iteration only occurs once, at initialization of the toolbox. Found results are stored and returned on subsequent
  calls.
  """
  global _imageTypes
  if _imageTypes is None:  # On first cal, enumerate possible input image types (original and any filters)
    _imageTypes = [member[3:-5] for member in dir(imageoperations)
                   if member.startswith('get') and member.endswith("Image")]

  return _imageTypes


def getTestCase(testCase, dataDirectory=None):
  """
  This function provides an image and mask for testing PyRadiomics. One of seven test cases can be selected:

   - brain1
   - brain2
   - breast1
   - lung1
   - lung2
   - test_wavelet_64x64x64
   - test_wavelet_37x37x37

  Checks if the test case (consisting of an image and mask file with signature <testCase>_image.nrrd and
  <testCase>_label.nrrd, respectively) is available in the ``dataDirectory``. If not available, the testCase is
  downloaded from the GitHub repository and stored in the ``dataDirectory``. Also creates the ``dataDirectory`` if
  necessary.
  If no ``dataDirectory`` has been specified, PyRadiomics will use a temporary directory: <TEMPDIR>/pyradiomics/data.

  If the test case has been found or downloaded successfully, this function returns a tuple of two strings:
  ``(path/to/image.nrrd, path/to/mask.nrrd)``. In case of an error ``(None, None)`` is returned.

  .. note::
    To get the testcase with the corresponding single-slice label, append "_2D" to the testCase.

  """
  global logger, testCases
  label2D = False
  testCase = testCase.lower()
  if testCase.endswith('_2d'):
    label2D = True
    testCase = testCase[:-3]

  if testCase not in testCases:
    raise ValueError('Testcase "%s" not recognized!' % testCase)

  logger.debug('Getting test case %s', testCase)

  if dataDirectory is None:
    dataDirectory = os.path.join(tempfile.gettempdir(), 'pyradiomics', 'data')
    logger.debug('No data directory specified, using temporary directory "%s"', dataDirectory)

  im_name = '%s_image.nrrd' % testCase
  ma_name = '%s_label%s.nrrd' % (testCase, '_2D' if label2D else '')

  def get_or_download(fname):
    target = os.path.join(dataDirectory, fname)
    if os.path.exists(target):
      logger.debug('File %s already downloaded', fname)
      return target

    # Test case file not found, so try to download it
    logger.info("Test case file %s not available locally, downloading from github...", fname)

    # First check if the folder is available
    if not os.path.isdir(dataDirectory):
      logger.debug('Creating data directory: %s', dataDirectory)
      os.makedirs(dataDirectory)

    # Download the test case files (image and label)
    url = r'https://github.com/Radiomics/pyradiomics/releases/download/v1.0/%s' % fname

    logger.debug('Retrieving file at %s', url)
    _, headers = urllib.request.urlretrieve(url, target)

    if headers.get('status', '') == '404 Not Found':
      raise ValueError('Unable to download image file at %s!', url)

    logger.info('File %s downloaded', fname)
    return target

  logger.debug('Getting Image file')
  imageFile = get_or_download(im_name)

  logger.debug('Getting Mask file')
  maskFile = get_or_download(ma_name)

  return imageFile, maskFile


def getParameterValidationFiles():
  """
  Returns file locations for the parameter schema and custom validation functions, which are needed when validating
  a parameter file using ``PyKwalify.core``.
  This functions returns a tuple with the file location of the schema as first and python script with custom validation
  functions as second element.
  """
  dataDir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'schemas'))
  schemaFile = os.path.join(dataDir, 'paramSchema.yaml')
  schemaFuncs = os.path.join(dataDir, 'schemaFuncs.py')
  return schemaFile, schemaFuncs


class _DummyProgressReporter(object):
  """
  This class represents the dummy Progress reporter and is used for where progress reporting is implemented, but not
  enabled (when the progressReporter is not set or verbosity level > INFO).

  PyRadiomics expects that the _getProgressReporter function returns an object that takes an iterable and 'desc' keyword
  argument at initialization. Furthermore, it should be iterable, where it iterates over the iterable passed at
  initialization and it should be used in a 'with' statement.

  In this class, the __iter__ function redirects to the __iter__ function of the iterable passed at initialization.
  The __enter__ and __exit__ functions enable usage in a 'with' statement
  """
  def __init__(self, iterable=None, desc='', total=None):
    self.desc = desc  # A description is not required, but is provided by PyRadiomics
    self.iterable = iterable  # Iterable is required

  def __iter__(self):
    return self.iterable.__iter__()  # Just iterate over the iterable passed at initialization

  def __enter__(self):
    return self  # The __enter__ function should return itself

  def __exit__(self, exc_type, exc_value, tb):
    pass  # Nothing needs to be closed or handled, so just specify 'pass'

  def update(self, n=1):
    pass  # Nothing needs to be updated, so just specify 'pass'


def getProgressReporter(*args, **kwargs):
  """
  This function returns an instance of the progressReporter, if it is set and the logging level is defined at level INFO
  or DEBUG. In all other cases a dummy progress reporter is returned.

  To enable progress reporting, the progressReporter variable should be set to a class object (NOT an instance), which
  fits the following signature:

  1. Accepts an iterable as the first positional argument and a keyword argument ('desc') specifying a label to display
  2. Can be used in a 'with' statement (i.e. exposes a __enter__ and __exit__ function)
  3. Is iterable (i.e. at least specifies an __iter__ function, which iterates over the iterable passed at
     initialization).

  It is also possible to create your own progress reporter. To achieve this, additionally specify a function `__next__`,
  and have the `__iter__` function return `self`. The `__next__` function takes no arguments and returns a call to the
  `__next__` function of the iterable (i.e. `return self.iterable.__next__()`). Any prints/progress reporting calls can
  then be inserted in this function prior to the return statement.
  """
  global handler, progressReporter
  if progressReporter is not None and logging.NOTSET < handler.level <= logging.INFO:
    return progressReporter(*args, **kwargs)
  else:
    return _DummyProgressReporter(*args, **kwargs)
  
def checkGpuAvailability():
    """
    Check if CUDA and GPU are available for computation.
    
    In a Google Colab environment (or other containerized environments), 
    'lspci' may not show the GPU. Instead, we use 'nvidia-smi' to check 
    for the presence of an NVIDIA GPU, and 'nvcc --version' to check for the 
    CUDA toolkit.
    
    Returns:
      bool: True if NVIDIA GPU and CUDA toolkit are available, False otherwise.
    """
    print("Checking GPU availability...")
    has_gpu = False
    has_cuda_tk = False

    # Check for NVIDIA GPU using nvidia-smi (works well in Colab)
    try:
        gpu_check = subprocess.run(
            'nvidia-smi',
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )
        if gpu_check.returncode == 0 and "NVIDIA" in gpu_check.stdout:
            has_gpu = True
            print("NVIDIA GPU found via nvidia-smi.")
        else:
            print("NVIDIA GPU not found via nvidia-smi.")
    except Exception as e:
        print(f"Error checking for GPU via nvidia-smi: {e}")

    # Check for CUDA toolkit (nvcc)
    try:
        cuda_check = subprocess.run(
            'nvcc --version',
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )
        if cuda_check.returncode == 0 and "cuda compilation tools" in cuda_check.stdout.lower():
            has_cuda_tk = True
            print("CUDA Toolkit (nvcc) found.")
        else:
            print("CUDA Toolkit (nvcc) not found or 'nvcc --version' failed.")
    except Exception as e:
        print(f"Error checking for CUDA toolkit via nvcc: {e}")

    final_check = has_gpu and has_cuda_tk
    print(f"GPU Availability Check Result: {final_check} (GPU: {has_gpu}, CUDA Toolkit: {has_cuda_tk})")
    return final_check


progressReporter = None

# 1. Set up logging
debugging = True
logger = logging.getLogger(__name__)  # 'radiomics'
logger.setLevel(logging.INFO)  # Set default level of logger to INFO to reflect most common setting for a log file

# Set up a handler to print out to stderr (controlled by setVerbosity())
handler = logging.StreamHandler()
# formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%Y-%m-%d %H:%M")  # Alternative format
formatter = logging.Formatter("%(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
# force level=WARNING for stderr handler, in case logging default is set differently (issue 102)
setVerbosity(logging.WARNING)

# 2. Define the available test cases
testCases = ('brain1', 'brain2', 'breast1', 'lung1', 'lung2', 'test_wavelet_64x64x64', 'test_wavelet_37x37x37')

# 3. Attempt to load and enable the C extensions.

# First, try to load CUDA extensions if GPU seems available
gpu_available = checkGpuAvailability()

cMatrices = None  # set cMatrices to None to prevent an import error in the feature classes.
cShape = None
try:
    from radiomics import _cmatrices as cMatrices  # noqa: F401
    from radiomics import _cshape as cShape  # noqa: F401

    cShape.calculate_coefficients =

except ImportError as e:
    if os.path.isdir(os.path.join(os.path.dirname(__file__), '..', 'data')):
        # It looks like PyRadiomics is run from source (in which case "setup.py develop" must have been run)
        logger.critical('Apparently running from root, but unable to load C extensions... '
                        'Did you run "python setup.py build_ext --inplace"?')
        raise Exception('Apparently running from root, but unable to load C extensions... '
                        'Did you run "python setup.py build_ext --inplace"?')
    else:
        logger.critical('Error loading C extensions', exc_info=True)
        raise e


#
# if gpu_available: # or True: # for DEBUGING TODO delete
#     logger.debug("GPU potentially available. Trying to import CUDA C extensions...")
#     try:
#         from radiomics.cuda import _cshape as cShape_cuda
#         # from radiomics.cuda import _cmatrices as cMatrices_cuda # If you have this
#
#         # Assign the cuda module to cShape
#         cShape = cShape_cuda
#         # Add attributes with the names expected by shape.py, pointing to the _cuda versions
#         cShape.calculate_coefficients = cShape_cuda.calculate_coefficients_cuda
#         cShape.calculate_coefficients2D = cShape_cuda.calculate_coefficients2D_cuda
#
#         from radiomics import _cmatrices as cMatrices # Assuming you always have CPU cMatrices TODO : IS THAT OKAY???
#     except ImportError:
#         logger.warning('GPU available, but CUDA extensions not found or failed to import. '
#                        'Did you build with BUILD_CUDA_EXTENSIONS=1? Falling back to CPU.', exc_info=False) # Hide traceback for cleaner log
#         logger.debug('CUDA Import Error:', exc_info=True) # Log full error for debugging
#         _gpu_available_and_used = False # Ensure flag is False
#         cShape = None # Ensure cShape is None so CPU fallback occurs
#     except AttributeError as e:
#          logger.error('GPU available, but CUDA module is missing expected attributes (_cuda suffix funcs?). '
#                       'Falling back to CPU.', exc_info=True)
#          _gpu_available_and_used = False
#          cShape = None # Ensure cShape is None so CPU fallback occurs
#
#     except Exception as e:
#         logger.error('GPU available, but an unexpected error occurred importing CUDA C extensions. '
#                      'Falling back to CPU.', exc_info=True)
#         _gpu_available_and_used = False # Ensure flag is False
#         cShape = None # Ensure cShape is None so CPU fallback occurs


# If CUDA wasn't loaded or GPU not available, load CPU extensions
# This block now correctly handles fallback if cShape is None after CUDA attempt
# if cShape is None:
#     logger.debug("Attempting to load CPU C extensions...")
#     try:
#         # Import CPU versions
#         from radiomics import _cshape as cShape_cpu
#         from radiomics import _cmatrices as cMatrices_cpu # Assuming CPU always has both
#
#         # Assign CPU modules - they should already have the correct function names
#         cShape = cShape_cpu
#         # Only assign cMatrices if it wasn't loaded by CUDA block
#         if cMatrices is None:
#             cMatrices = cMatrices_cpu
#             logger.info('Using CPU C extensions for matrix calculations.')
#         else:
#              logger.info('Using CPU C extensions for shape, GPU (if loaded) for matrices.') # Adjust if needed
#
#         if gpu_available and not _gpu_available_and_used:
#              # Log only if GPU was available but we ended up using CPU
#              logger.info('Falling back to CPU C extensions for shape calculations.')
#         elif not gpu_available:
#              logger.info('Using CPU C extensions for shape calculations (GPU not available or not selected).')
#         # Else: GPU available and used (handled above)
#
#     except ImportError:
#         logger.critical('Failed to load critical CPU C extensions (_cshape)! PyRadiomics features requiring this may fail.', exc_info=True)
#         # Keep cShape as None
#         # Load CPU cMatrices if possible, even if cShape failed?
#         if cMatrices is None:
#             try:
#                 from radiomics import _cmatrices as cMatrices_cpu
#                 cMatrices = cMatrices_cpu
#                 logger.info("Loaded CPU C extensions for matrices, but shape failed.")
#             except ImportError:
#                  logger.critical('Failed to load CPU C extensions for matrices as well.', exc_info=True)
#                  cMatrices = None # Ensure it's None
#         # Decide whether to raise an error or allow running without C extensions
#         # raise ImportError('Failed to load required C extensions.') # Option 1: Fail hard
#         # cShape = None # Option 2: Allow running (but shape features will fail later) - Already None
#         # cMatrices = None # Already handled
#     except Exception as e:
#         logger.critical('Unexpected error loading CPU C extensions!', exc_info=True)
#         cShape = None # Ensure None on error
#         cMatrices = None # Ensure None on error

# if cShape is None:
#     logger.error("Critical C extension '_cshape' could not be loaded (neither CPU nor GPU). "
#                    "Shape features will not be available and will raise errors.")
#     # Keep cMatrices status separate if needed
#     if cMatrices is None:
#         logger.warning("C extension '_cmatrices' also failed to load. Performance may be suboptimal.")

# try:
#     from radiomics import _cmatrices as cMatrices
#     from radiomics import _cshape as cShape
# except ImportError:
#     cMatricesCpu = None
#     cShapeCpu = None

# 4. Enumerate features and image types (keep existing)
_featureClasses = None
_imageTypes = None
getFeatureClasses()
getImageTypes()

# 5. Set version (keep existing)
__version__ = get_versions()['version']
del get_versions

logger.info(f"PyRadiomics configuration: {'GPU' if gpu_available else 'CPU'} computation; Version {__version__}")
