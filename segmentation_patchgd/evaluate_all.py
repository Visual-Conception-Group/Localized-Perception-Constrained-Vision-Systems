# data_name = "aerial"
data_name = "drive"

model = "unet"
# model = "dlab"

base_folder = "/data/home/vinayv/codes/OUTPUT_Models"
main_folder = f"{base_folder}/{data_name}/{model}/*_*"


import os 
from glob import glob
from evaluate_models import evaluate_model
folders = glob(main_folder)
from main_config import CFG


for folder in folders: 
    cfg_path = None
    m1_path = None
    m2_path = None

    for cfg_path in glob(f"{folder}/config*"): pass
    for m1_path in glob(f"{folder}/*m1*best*"): pass
    for m2_path in glob(f"{folder}/*m2*best*"): pass

    print(folder)
    model_type = folder.split("/")[-1][2:]
    print("cfg:",cfg_path)
    print("m1:",m1_path)
    print("m2:",m2_path)
    print("model type:", model_type)
    print()

    config_path = "config.yaml"
    config = CFG(config_path)

    config.model_type = model_type
    config.model = model
    config.data_name = data_name
    config.model1_path = m1_path
    config.model2_path = m2_path
    config.exp_name = f"{data_name}_{model}_{os.path.basename(folder)}"
    config.update()
    print(config.train_data_root)

    try:
        evaluate_model(config)
    except Exception as e:
        raise e
        # print(f"===ERROR - {e} - {cfg_path, m1_path, m2_path}")

    # exit()


