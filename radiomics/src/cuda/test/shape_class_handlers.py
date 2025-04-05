import os
import datetime
import numpy as np

def write_shape_class_to_file(shapeClass, base_dir=None):
  """
  Write shape class atributes to separate files in a date-based folder.
  
  Args:
    shapeClass: The shape class instance containing feature values
    base_dir: Base directory to create the date folder in (default: current directory)
  """
  try:
    # Create folder with current date and time including seconds
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if base_dir:
      output_dir = os.path.join(base_dir, "data", now)
    else:
      output_dir = os.path.join("data", now)
    # Ensure the directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Write 3 text files for different features
    with open(os.path.join(output_dir, "surface_area.txt"), 'w') as f:
      f.write(f"{shapeClass.SurfaceArea}")
    
    with open(os.path.join(output_dir, "volume.txt"), 'w') as f:
      f.write(f"{shapeClass.Volume}")
    
    with open(os.path.join(output_dir, "diameters.txt"), 'w') as f:
      f.write(f"{shapeClass.diameters}")
    
    # Save pixel_spacing as binary NumPy file instead of text
    np.save(os.path.join(output_dir, "pixel_spacing.npy"), shapeClass.pixelSpacing)
    
    # Write mask array as binary file
    np.save(os.path.join(output_dir, "mask_array.npy"), shapeClass.maskArray)
    
    print(f"Shape features successfully written to {output_dir}")
  except Exception as e:
    print(f"Error writing shape features to files: {e}")

def load_shape_class(datetime_folder, base_dir=None):
  """
  Load shape class atributes values from files in a datetime-based folder.
  
  Args:
    datetime_folder: The datetime folder name in format 'YYYY-MM-DD_HH-MM-SS'
    base_dir: Base directory where the datetime folder is located (default: current directory)
    
  Returns:
    dict: Dictionary containing the shape features (surface_area, volume, diameters, pixel_spacing, mask_array)
  """
  print("dsa'")

  try:
    # Determine the full path to the datetime folder
    if base_dir:
      folder_path = os.path.join(base_dir, "data", datetime_folder)
    else:
      folder_path = os.path.join("data", datetime_folder)
    
    # Check if folder exists
    if not os.path.exists(folder_path):
      raise FileNotFoundError(f"Folder {folder_path} does not exist")
    
    # Dictionary to store results
    features = {}
    
    # Read surface area
    with open(os.path.join(folder_path, "surface_area.txt"), 'r') as f:
      features['surface_area'] = float(f.read())
    
    # Read volume
    with open(os.path.join(folder_path, "volume.txt"), 'r') as f:
      features['volume'] = float(f.read())
    
    # Read diameters
    with open(os.path.join(folder_path, "diameters.txt"), 'r') as f:
      # Convert string representation of list to actual list
      diameters_str = f.read().strip()
      # Handle the format, which might be like "[1.0, 2.0, 3.0]"
      features['diameters'] = eval(diameters_str)
    
    # Load pixel spacing using np.load instead of text parsing
    pixel_spacing_path = os.path.join(folder_path, "pixel_spacing.npy")
    features['pixel_spacing'] = np.load(pixel_spacing_path)
    
    # Load mask array
    mask_path = os.path.join(folder_path, "mask_array.npy")
    features['mask_array'] = np.load(mask_path)
    
    print(f"Shape features successfully loaded from {folder_path}")
    return features
    
  except Exception as e:
    print(f"Error loading shape features from files: {e}")
    return None



