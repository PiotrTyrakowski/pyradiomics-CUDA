LABEL_MAPPING = {
    1: "aorta",
    2: "left_lung",
    3: "right_lung",
    4: "trachea",
    5: "pulmonary_artery",
}
MASK_PATH = "./masks.nii.gz"
SCAN_PATH = "./scan.nii.gz"
CONFIG_PATH = "./extractor_config.json"


from radiomics.featureextractor import RadiomicsFeatureExtractor
import os
print("Current working directory:", os.getcwd())
os.chdir('radiomics/cuda/biecek_test')
print("Current working directory after change:", os.getcwd())
print("Files in current directory:", os.listdir())

extractor = RadiomicsFeatureExtractor()
extractor.loadParams(CONFIG_PATH)

features = {}
for val, name in LABEL_MAPPING.items():
    features[name] = extractor.execute(SCAN_PATH, MASK_PATH, val)
    break