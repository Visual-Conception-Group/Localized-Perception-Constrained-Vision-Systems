from src.data.data import load_datasets, transformations
from src.models.patchgd_models import Identity, z_block_v1 as z_block
import torch 
import torch.nn as nn
from src.utils import (
    load_checkpoint,
)
import os 

"""
load_data
get_models
eval_load_checkpoint
"""

def load_data(config):
    # LOADING DATASET
    train_path = f"{config.train_data_root}/{config.train_folder}"
    valid_path =  f"{config.train_data_root}/{config.valid_folder}"

    print("==> Data Loading Started")
    train_loader, valid_loader = load_datasets(
        train_path,
        valid_path,
        batch_size=config.batch_size,
        img_size=config.ip_size,
        n_samples=5, 
        transforms=transformations,
        n_patch_split=config.data_split_patch_side,
    )

    use_dummy_data = False
    # DUMMY DATA LOADER, IF DATA NOT PRESENT
    if use_dummy_data:
        x = torch.rand((4, 3, 512, 512))
        y = torch.rand((4, 1, 512, 512))
        loader = [(x,y), (x,y), (x,y), (x,y)]
        train_loader = loader
        valid_loader = loader
    # exit()

    print(f"Data Loader: {len(train_loader), len(valid_loader)} | Data Split: {config.data_split_patch_side}")
    
    if len(train_loader)==0 or len(valid_loader)==0:
        print("==> Data Not Loaded")
        exit()
    else:
        print("==> Data Loaded Successfully")

    return train_loader, valid_loader

def get_models(config):
    model1, model2 = None, None
    n_classes = 1

    if config.model_type in ["full", "down", "tiled"]:
        model2 = Identity()

        if config.model == "unet":
            from src.models.base_unet_model import build_unet
            model1 = build_unet()
            model1.outputs = nn.Conv2d(64, n_classes, kernel_size=1, padding=0) # reducing the channels
        if config.model == "dlab":
            model1 = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=False)
            model1.classifier[4] = nn.Conv2d(256, n_classes, kernel_size=1, padding=0)
        optimizer1 = torch.optim.AdamW(model1.parameters(), lr=1e-4)
        optimizer2 = None

    if config.model_type in ["pgd", "pgd_global"]:
        if config.model_type == "pgd_global":
            print("Using Global Features")
            if config.global_method=='cat':
                model2 = z_block(2*config.feature_size, 1)
            elif config.global_method=='add':
                model2 = z_block(config.feature_size, 1)
            else:
                print("Concat mode not specified"); exit()
        else:
            model2 = z_block(config.feature_size, 1)

        if config.model == "unet":
            from src.models.base_unet_model import build_unet
            model1 = build_unet()
            # model1.outputs = Identity() # if not already identity
            model1.outputs = nn.Conv2d(64, config.feature_size, kernel_size=1, padding=0) # reducing the channels
        if config.model == "dlab":
            model1 = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=False)
            model1.classifier[4] = nn.Conv2d(256, config.feature_size, kernel_size=1, padding=0) # reducing the channels

        optimizer1 = torch.optim.AdamW(model1.parameters(), lr=1e-4)
        optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-4)


    print(model1)
    print(model2)
    return model1, model2, optimizer1, optimizer2

def eval_load_checkpoint(config, model1, model2):
    # LOAD CHECKPOINT
    # MODEL 1
    if os.path.exists(config.model1_path):
        model1 = load_checkpoint(config.model1_path, model1)
        print(f"Checkpoint M1 Loaded: {config.model1_path}")
    else:
        print(f"Checkpoint M1 not Found at: {config.model1_path}")
        # exit()


    # MODEL 2
    if config.model2_path is not None and os.path.exists(config.model2_path):
        model2 = load_checkpoint(config.model2_path, model2)
        print(f"Checkpoint M2 Loaded: {config.model2_path}")
    else:
        print(f"Checkpoint M2 not Found at: {config.model2_path}")
        # exit()

    return model1, model2