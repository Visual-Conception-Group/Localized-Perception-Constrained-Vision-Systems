from src.models.patchgd_models import Identity, z_block_v1 as z_block
from src.models.base_unet_model import build_unet
from src.models.loss import DiceBCELoss

from src.utils import (
    create_dir,
    run_test_on_model,
    plot_graph,
    epoch_time,
    load_checkpoint,
)

from src.patchgd.patchgd_utils import fill_z, merge_patches, sample_k_patches, combine_global

from common_utils import load_data, get_models, eval_load_checkpoint
import torch 
import time
from tqdm.auto import tqdm 
import os
import cv2
import numpy as np
import torchvision


def new_print(*args):
    for a in args:
        tqdm.write(str(a), end=" ")
    tqdm.write("")
print = new_print

"""
LOADING CONFIG
LOADING DATASET
LOADING MODEL AND CHECKPOINT
train()
- train_one_epoch_patchgd()
- train_one_epoch_singlemodel()
evaluate()
run_on_images()
infer_model()
- get_per_patch_prediction()
"""

from main_config import CFG
config_path = "config.yaml"
config = CFG(config_path)
print("\n<============ CONFIG LOADED ============>\n\n\n")

################################[ DATA LOADING ]################################
save_files = True
create_new_dir = True
# save_files = False
# create_new_dir = False

train_loader, valid_loader = load_data(config)

## CREATING DIRECTORIES
checkpoint_name = f"ckpt_{config.patch_name}"
op_dir = config.op_dir
model_op_dir = f"{op_dir}/weights"

if create_new_dir: create_dir(op_dir); create_dir(model_op_dir)

import shutil
config_op = os.path.join(op_dir, os.path.basename(config_path))
shutil.copy(src=config_path,dst=config_op)
print("Config Copied to: ", config_path, config_op)
################################[ DATA LOADING ]################################

################################[ MODEL LOADING ]###############################

model1, model2, optimizer1, optimizer2 = get_models(config)

loss_fn = DiceBCELoss()
device = torch.device('cuda')   ## RTX 3090
model1.to(device)
model2.to(device)

# LOAD CHECKPOINT
model1, model2 = eval_load_checkpoint(config, model1, model2)
# exit()
################################[ MODEL LOADING ]###############################


################################[ TRAINING ]####################################
# with inner iteration
def train_one_epoch_patchgd(loader, epoch_str=""):
    start_time = time.time()
    epoch_loss = 0.0
    start = True
    size = len(loader)
    print("::Loader Size", size)

    print("**** In Train Mode ****")
    model1.train()
    for idx, (x, y) in enumerate(tqdm(loader, desc=f'Epoch {epoch_str}', position=0)):
        x, y = x.to(device, dtype=torch.float32), y.to(device, dtype=torch.float32) # can be batched image
        # print("Input:", x.shape, y.shape)
        img_b, img_c, img_h, img_w  = x.shape

        # CREATING Z MATRIX
        with torch.no_grad():
            # print("==> Getting Z without Gradients")
            b_img_patches, org_img_patches_feats = fill_z(model1, x, config.patch_side, feature_size=config.feature_size) # batch of images

        if start:
            start = False
            print(f"Patch Size - {b_img_patches.shape}, Patch Feat Size - {org_img_patches_feats.shape}")
        # 10 sec
        """
        It will fail on batches of size 1 if we use feature-wise batch normalization.
        As Batch normalization computes:
        y = (x - mean(x)) / (std(x) + eps)
        If we have one sample per batch then mean(x) = x, and the output will be entirely zero (ignoring the bias). We can't use that for learning...
        """
        with torch.no_grad():
            if config.global_feat:
                resizedImage = torchvision.transforms.Resize( size=(config.patch_size, config.patch_size))(x)
                model1.eval()
                global_feat = model1(resizedImage)['out']
                model1.train()
                global_feat_resized = torchvision.transforms.Resize(size=(config.ip_size, config.ip_size))(global_feat)
                print("==> GLOBAL:: ", resizedImage.shape, global_feat.shape, global_feat_resized.shape)

        # 11 sec
        # INNER ITERATIONS
        optimizer1.zero_grad()
        optimizer2.zero_grad()


        for j_iter in range(config.n_inner_iter):
            print(f"            Inner Iteration: {j_iter}")
            b_img_patches_feats = org_img_patches_feats.detach().clone()

            # UPDATING REPRESENTATION FOR EACH IMAGE
            for b in range(img_b):
                k_img_patches, positions = sample_k_patches(
                    b_img_patches[b],
                    k=config.k_patches,
                    patch_side=config.patch_side
                )
                # print(f"Sampled Patches")
                k_img_patches = k_img_patches.to(device)

                # GENERATING FEATURES OF K PATCHES ONLY 
                k_features = model1(k_img_patches)['out']
                # print(f"---- K Features: {x.shape}, {y.shape} || Pred Shape: {k_features.shape}")

                # REPLACING K PATCHES FOR EACH IMAGE
                for kth_patch, (i_x,i_y) in enumerate(positions):
                    # b_img_patches_feats[b][i_x][i_y] = torch.ones(k_features[kth_patch].shape) # NO GRAD TO TRAIN
                    # print("patch positions:", i_x,i_y)
                    b_img_patches_feats[b][i_x][i_y] = k_features[kth_patch]
            
            # continue
            
            # CREATING Z MATRIX FROM FEATURES
            batch_z_matrix = merge_patches(b_img_patches_feats, patch_side=config.patch_side)
            

            # PLOTTING THE PATCHES
            # print("------------- Z Mean - ", batch_z_matrix.shape)
            # for z_idx, batch_z_matrix_id in enumerate(batch_z_matrix):
            #     mean_img = get_zmetrics_mean(torch.unsqueeze(batch_z_matrix_id, dim=0))
            #     cv2.imwrite(f"mean_imgs/mean_img_{j_iter}_{idx}_{z_idx}.png", mean_img*255)
            # exit()
            
            # HAVE TO RESIZE AGAIN
            # input("=============== Model 1 Till now ===============")

            # Final Matrix in batch
            batch_z_matrix = batch_z_matrix.to(device)
            
            if config.global_feat:
                batch_z_matrix = combine_global(batch_z_matrix, global_feat_resized, method=config.global_method)
            
            # print("z matrix::", batch_z_matrix.shape)
            # input("waiting for model 2")

            y_pred = model2(batch_z_matrix)

            # print("y pred:", y_pred.shape, y.shape)
            loss = loss_fn(y_pred, y) / config.n_acc_steps
            loss.backward(retain_graph=True)

        
            if ((j_iter + 1) % config.n_acc_steps == 0) or (j_iter+1 == config.n_inner_iter):
                print(f"--> Updating the weights: ({j_iter+1}/{config.n_inner_iter})")
                optimizer1.step()
                if not optimizer2 is None: optimizer2.step()
                optimizer1.zero_grad()
                if not optimizer2 is None: optimizer2.zero_grad()

        # CREATING Z MATRIX
        with torch.no_grad():
            # print("==> [Eval] Getting Z without Gradients")
            b_img_patches, b_img_patches_feats = fill_z(model1, x, config.patch_side, feature_size=config.feature_size) # batch of images

        # CREATING Z MATRIX FROM FEATURES
        batch_z_matrix = merge_patches(b_img_patches_feats, patch_side=config.patch_side)

        # global_feat_resized.to(device)
        batch_z_matrix = batch_z_matrix.to(device)

        if config.global_feat:
            print("Dev batchz:", batch_z_matrix.get_device())
            print("Dev global feat res:", global_feat_resized.get_device())
            batch_z_matrix = combine_global(batch_z_matrix, global_feat_resized, method=config.global_method)

        print("Final z matrix::", batch_z_matrix.shape)
        y_pred = model2(batch_z_matrix)

        loss = loss_fn(y_pred, y)
        epoch_loss += loss.item()

    # print(f"---- Last Batch Input Shape: {x.shape}, {y.shape} || Pred Shape: {y_pred.shape}")
    epoch_loss = epoch_loss/size
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    print(f"Epoch Time: {epoch_mins}m {epoch_secs}s")

    return epoch_loss   

# without inner iterations
def train_one_epoch_single_model(loader, epoch_str=""):
    start_time = time.time()
    epoch_loss = 0.0
    size = len(loader)
    print("::Loader Size", size)
    start = True

    print("**** In Train Mode")
    model1.train()
    for x, y in tqdm(loader):
        if len(x)==1:
            continue

        x = x.to(device, dtype=torch.float32) # can be batched image
        y = y.to(device, dtype=torch.float32)

        
        optimizer1.zero_grad()
        y_pred = model1(x)['out']
        
        if start:
            start = False
            print(f"Input Size - {x.shape, y.shape}, Pred Feat Size - {y_pred.shape, y.shape}")
        

        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer1.step()

        epoch_loss += loss.item()

    print(f"---- Last Batch Input Shape: {x.shape}, {y.shape} || Pred Shape: {y_pred.shape}")
    epoch_loss = epoch_loss/size
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    # print(f"Epoch Time: {epoch_mins}m {epoch_secs}s")
    return epoch_loss   

def evaluate(loader):
    """
    it gets the broken image as parts for evaluation if there. 
    same batch size as training
    """
    epoch_loss = 0.0
    size = len(loader)
    print("::Loader Size", size)

    print("**** In Eval Mode")
    model1.eval()
    for x, y in tqdm(loader):
        x = x.to(device, dtype=torch.float32) # can be batched image
        y = y.to(device, dtype=torch.float32)

        with torch.no_grad():
            
            # y_pred = model1(img) # [1, 1, 512, 512] # DEFAULT INFERENCE OF SINGLE IMAGE
            Z_mat = get_per_patch_prediction(
                model=model1,
                inputs=x,
                patch_side=config.patch_side,
                feature_size=config.feature_size,
            )
            Z_mat = Z_mat.to(device)

            if config.model_type == "pgd_global":
                if config.global_feat:
                    resizedImage = torchvision.transforms.Resize( size=(config.patch_size, config.patch_size))(x)
                    global_feat = model1(resizedImage)['out']
                    global_feat_resized = torchvision.transforms.Resize( size=(config.ip_size, config.ip_size))(global_feat)
                    Z_mat = combine_global(Z_mat, global_feat_resized, method=config.global_method)

            y_pred = model2(Z_mat)
            loss = loss_fn(y_pred, y)

        epoch_loss += loss.item()

    # print(f"---- Last Batch Input Shape: {x.shape}, {y.shape} || Pred Shape: {y_pred.shape}")
    epoch_loss = epoch_loss/size
    return epoch_loss   


def train():
    train_losses, valid_losses, best_valid_loss, best_epoch = [], [], float("inf"), 0
    train_loss, valid_loss = 0, 0

    val_interval = config.val_interval
    n_epoch = config.num_epoch

    for epoch in range(n_epoch):
        torch.cuda.empty_cache()

        print(f"training the model [{epoch+1}/{n_epoch}]")
        if config.model_type in ["full", "down", "tiled"]:
            print(["full", "down", "tiled"])
            train_loss = train_one_epoch_single_model(train_loader, epoch_str=f"[{epoch+1}/{n_epoch}]")
        
        elif config.model_type in ["pgd", "pgd_global"]:
            print(["pgd", "pgd_global"])
            train_loss = train_one_epoch_patchgd(train_loader, epoch_str=f"[{epoch+1}/{n_epoch}]")

        valid_loss = evaluate(valid_loader)
        if epoch % val_interval==0:
            if save_files:
                run_on_images(epoch)
                # SAVING EVERY EPOCH MODEL
                if optimizer1 is not None: torch.save(model1.state_dict(), f"{model_op_dir}/m1_{checkpoint_name}_{epoch}_{valid_loss}.pth")
                if optimizer2 is not None: torch.save(model2.state_dict(), f"{model_op_dir}/m2_{checkpoint_name}_{epoch}_{valid_loss}.pth")

        data_str = f'Epoch: {epoch+1:02} |\n \tTrain Loss: {train_loss:.3f}\n \tVal. Loss: {valid_loss:.3f}\n'
        print(data_str)
        
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        plot_graph(train_losses, valid_losses, path=f"{op_dir}/train_logs_{checkpoint_name}.png")

        # SAVING BEST MODEL
        if valid_loss < best_valid_loss:
            data_str = f"Valid loss improved from {best_valid_loss:2.4f} to {valid_loss:2.4f}. Saving checkpoint: {checkpoint_name}"
            print(data_str)
            best_valid_loss = valid_loss
            best_epoch = epoch

            if save_files:
                if optimizer1 is not None: torch.save(model1.state_dict(), f"{op_dir}/m1__{checkpoint_name}_best.pth")
                if optimizer2 is not None: torch.save(model2.state_dict(), f"{op_dir}/m2__{checkpoint_name}_best.pth")
        else: 
            print(f"Not Saving the model {valid_loss} - best epoch:{best_epoch}, {best_valid_loss}")

    if save_files:
        # SAVING THE LAST MODEL
        if optimizer1 is not None: torch.save(model1.state_dict(), f"{op_dir}/last_m1_{checkpoint_name}_{epoch}_{valid_loss}.pth")
        if optimizer2 is not None: torch.save(model2.state_dict(), f"{op_dir}/last_m2_{checkpoint_name}_{epoch}_{valid_loss}.pth")
    print("MODEL TRAINING COMPLETED")


def get_per_patch_prediction(model, inputs, patch_side, feature_size, show_lines=False):
    # BREAK IMAGE INTO PATCHES
    # GET FEATURES OF EACH PATCH
    # MERGE IT TO CREATE A IMAGE
    # IMAGE PATCH WILL FIT IN GPU, BY SUB-BATCHING
    import torch.nn.functional as F
    import torchvision

    # 1. GENERATE FEATURE FOR EACH PATCH
    b_img_patches, b_img_patches_feats = fill_z(
        model = model,
        inputs = inputs, 
        patch_side = patch_side,
        feature_size = feature_size,
    )

    # 2. TO VISUALIZE INDIVIDUAL PATCH MERGE
    def add_lines(feat):
        img_size = feat.shape[-1]
        out = torchvision.transforms.Pad(1, fill=1, padding_mode='constant')(feat)
        out = torchvision.transforms.Resize(size=img_size)(out)
        return out

    if show_lines:
        feats = b_img_patches_feats[0]
        for i in range(patch_side):
            for j in range(patch_side):
                print(f"[{i}, {j}], {feats[i][j].shape}")
                feats[i][j] = add_lines(feats[i][j])

    # 3. LOCALIZED JOINING OF FEATURES IN Z MATRIX
    y_pred = merge_patches(b_img_patches_feats, patch_side=config.patch_side)

    return y_pred

def run_on_images(epoch):
    result_folder = f"{op_dir}/epochs/_epoch{epoch}"
    # model2 = Identity()
    # model1 = model1
    # model2 = model2

    print("Global Feat:", config.global_feat)

    run_test_on_model(
        model1,
        model2,
        result_folder,
        sample          = 5,
        patch_side      = config.patch_side,
        feature_size    = config.feature_size,
        ip_size         = config.ip_size,
        gt_size         = config.gt_size,
        data_path       = config.eval_data_path,
        use_global_feat = config.global_feat,
        patch_size      = config.patch_size,
        method          = config.global_method
    )

def infer_model(img_path, op_path=None, write=False):
    """
    """
    ip_size = config.ip_size
    file = os.path.basename(img_path) 

    read_resize = lambda x, size: resize(read(x), size)
    resize = lambda x, size: cv2.resize(x, size)
    read = lambda x: cv2.imread(x)
    
    img = read_resize(img_path, ip_size)
    img = np.transpose(img, (2, 0, 1)) # channel first
    img = img/255.0 # normalizing image
    img = np.expand_dims(img, axis=0) # creating batch of image
    img = torch.from_numpy(img).float().to(device) # converting to tensor

    with torch.no_grad():
        # y_pred = model1(img) # [1, 1, 512, 512] # DEFAULT INFERENCE OF SINGLE IMAGE
        Z_mat = get_per_patch_prediction(
            model=model1,
            inputs=img,
            patch_side=config.patch_side,
            feature_size=config.feature_size,
            show_lines=True
        )

        # FULL AND DOWNSCALED WILL HAVE IDENTITY AS THETA2 - SO NO AFFECT
        Z_mat = Z_mat.to(device)
        y_pred = model2(Z_mat)

    y_pred = y_pred.detach().cpu().numpy()[0][0] # selecting first channel mask
    y_pred = np.array(y_pred > 0.5, dtype=np.float32) # thresholding
    y_pred = np.array(y_pred*255, dtype=np.uint8)

    if write:
        if op_path==None: op_path = f'{file[:-4]}_op_mask.png'

        flag = cv2.imwrite(op_path, y_pred)
        if flag: print(f"::- Mask written at {op_path}")
        else: print(f"::- Mask Not written at {op_path}")
        return y_pred, op_path
    return y_pred

################################[ TRAINING ]####################################

"""
ONLY TRAINING THE MODEL 2
IRRESPECTIVE OF MODEL 1

load the z model, in eval mode - get the matrix created 
define the model 2
infer it - train the model 2
"""
##########################################################################################
    
if __name__=="__main__":
    train()

    
    # img_path = "retina/test/image/0.png"
    # mask, op_path = infer_model(
    #     img_path, 
    #     op_path=f"{config.patch_side}_blur_testlines_patch_op.png",
    #     write=True
    #     )
