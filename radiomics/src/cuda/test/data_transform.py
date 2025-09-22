from __future__ import annotations

import argparse
import collections
import os

import numpy as np

from radiomics import getFeatureClasses, imageoperations
from radiomics.featureextractor import RadiomicsFeatureExtractor

def _write_shape_class_to_file(shapeClass, out_dir, base_dir=None):
    try:
        if base_dir:
            output_dir = os.path.join(base_dir, "data", out_dir)
        else:
            output_dir = os.path.join("data", out_dir)
        os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(output_dir, "surface_area.txt"), "w") as f:
            f.write(f"{shapeClass.SurfaceArea}")

        with open(os.path.join(output_dir, "volume.txt"), "w") as f:
            f.write(f"{shapeClass.Volume}")

        with open(os.path.join(output_dir, "diameters.txt"), "w") as f:
            f.write(f"{shapeClass.diameters}")

        np.save(os.path.join(output_dir, "pixel_spacing.npy"), shapeClass.pixelSpacing)

        np.save(os.path.join(output_dir, "mask_array.npy"), shapeClass.maskArray)

        print(f"Shape features successfully written to {output_dir}")

        return output_dir
    except Exception as e:
        print(f"Error writing shape features to files: {e}")

def load_shape_class(folder_path):
    try:

        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder {folder_path} does not exist")

        features = {}

        with open(os.path.join(folder_path, "surface_area.txt"), "r") as f:
            features["surface_area"] = float(f.read())

        with open(os.path.join(folder_path, "volume.txt"), "r") as f:
            features["volume"] = float(f.read())

        with open(os.path.join(folder_path, "diameters.txt"), "r") as f:
            diameters_str = f.read().strip()
            features["diameters"] = eval(diameters_str)

        pixel_spacing_path = os.path.join(folder_path, "pixel_spacing.npy")
        features["pixel_spacing"] = np.load(pixel_spacing_path)

        mask_path = os.path.join(folder_path, "mask_array.npy")
        features["mask_array"] = np.load(mask_path)

        print(f"Shape features successfully loaded from {folder_path}")
        return features

    except Exception as e:
        print(f"Error loading shape features from files: {e}")
        return None

class RadiomicsFeatureWriter(RadiomicsFeatureExtractor):
    def __init__(self, base_dir=None, prefix="data"):
        super().__init__()
        self.base_dir = base_dir
        self.prefix = prefix
        self.out_dirs = []
        self.idx = 0

    def saveShape(self, image, mask, boundingBox, **kwargs):
        featureVector = collections.OrderedDict()
        enabledFeatures = self.enabledFeatures
        croppedImage, croppedMask = imageoperations.cropToTumorMask(
            image, mask, boundingBox
        )

        Nd = mask.GetDimension()
        if "shape" in enabledFeatures.keys():
            if Nd != 3:
                raise RuntimeError("Shape features are only implemented for 3D images.")

            shapeClass = getFeatureClasses()['shape'](
                croppedImage, croppedMask, **kwargs
            )
            output = _write_shape_class_to_file(
                shapeClass, f"{self.prefix}_{self.idx}", self.base_dir
            )
            self.out_dirs.append(output)

        if "shape2D" in enabledFeatures.keys():
            raise NotImplementedError("2D shape features are not implemented yet.")

        return featureVector

    def save_npy_files(self, scan_path, mask_path, idx):
        old_shape_compute = self.computeShape
        self.idx = idx

        self.computeShape = self.saveShape
        result = self.execute(scan_path, mask_path, idx)
        self.computeShape = old_shape_compute

        return result

    def get_saved_dirs(self):
        return self.out_dirs

DEFAULT_CONFIG = """
{
    "imageType": {
        "Original": {}
    },
    "featureClass": {
        "shape": [
            "MeshVolume",
            "VoxelVolume",
            "SurfaceArea",
            "SurfaceVolumeRatio",
            "Maximum3DDiameter",
            "Maximum2DDiameterSlice",
            "Maximum2DDiameterColumn",
            "Maximum2DDiameterRow",
            "MajorAxisLength"
        ]
    },
    "setting": {
        "additionalInfo": false
    }
}
"""

def write_shape_class(
    mask_path, scan_path, max_idx, base_dir=None, config_path=None, prefix="data"
):
    extractor = RadiomicsFeatureWriter(base_dir, prefix)

    if config_path is None:
        config_path = os.path.join(os.curdir, "tmp_cfg.json")
        with open(config_path, "w") as f:
            f.write(DEFAULT_CONFIG)

    extractor.loadParams(config_path)

    for val in range(1, max_idx + 1):
        extractor.save_npy_files(scan_path, mask_path, val)

    return extractor.get_saved_dirs()

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Extract and save radiomics shape features from medical images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-m",
        "--mask-path",
        required=True,
        help="Path to the mask file",
    )

    parser.add_argument(
        "-s",
        "--scan-path",
        required=True,
        help="Path to the scan/image file",
    )

    parser.add_argument(
        "-n",
        "--max-idx",
        type=int,
        required=True,
        help="Maximum index for processing (will process indices 1 to max_idx)",
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="Base output directory for saving features (default: current directory)",
    )

    parser.add_argument(
        "-c",
        "--config-path",
        default=None,
        help="Path to custom configuration file (default: uses built-in config)",
    )

    parser.add_argument(
        "-p",
        "--prefix",
        default="data",
        help="Prefix for output folder names (will create folders like PREFIX_1, PREFIX_2, etc.)",
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    output_dirs = write_shape_class(
        mask_path=args.mask_path,
        scan_path=args.scan_path,
        max_idx=args.max_idx,
        base_dir=args.output_dir,
        config_path=args.config_path,
        prefix=args.prefix,
    )

    print("\nProcessing completed successfully!")
    print(f"Created {len(output_dirs)} output directories:")
    for dir_path in output_dirs:
        print(f"  - {dir_path}")
