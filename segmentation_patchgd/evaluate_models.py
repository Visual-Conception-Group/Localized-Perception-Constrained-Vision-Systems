from main_config import CFG as config
from common_utils import get_models, eval_load_checkpoint
import torch
from src.utils import (
    run_test_on_model,
    seeding
)
from common_utils import get_models


device = torch.device('cuda')   ## RTX 3090
def evaluate_model(config):
    model1, model2, optimizer1, optimizer2 = get_models(config)
    device = torch.device('cuda')
    model1.to(device)
    if model2 is not None: model2.to(device)
    
    model1, model2 = eval_load_checkpoint(config, model1, model2)
    run_on_images(model1, model2, config)


################################[ MODEL LOADING ]###############################

def run_on_images(model1, model2, config):
    seeding(42)
    exp_name = config.exp_name
    result_folder = f"OUTPUT/result_{exp_name}_{config.ip_size}"
    print("Global Feat:", config.global_feat)
    
    with torch.no_grad():
        run_test_on_model(
            model1,
            model2,
            result_folder,
            sample          = 200,
            patch_side      = config.patch_side,
            feature_size    = config.feature_size,
            device          = device,
            ip_size         = config.ip_size,
            gt_size         = config.gt_size,
            data_path       = config.eval_data_path, 
            use_global_feat = config.global_feat,
            method          = config.global_method,
            patch_size      = config.patch_size,
        )

if __name__ == "__main__":
    from main_config import CFG
    config_path = "config.yaml"
    config = CFG(config_path)

    try:
        evaluate_model(config)
    except Exception as e:
        raise e
        # print(f"===ERROR - {e} - {cfg_path, m1_path, m2_path}")
    evaluate_model(config_path)