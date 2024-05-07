import torch 
import numpy as np
from src.models.loss import DiceBCELoss, BCELoss, DiceOnly
from tqdm import tqdm
from torch.nn import functional as F
import torchvision
import time 

"""
combine_global
sample_k_patches - get k patches out of all patches
merge patches - create a image out of patches
batch_patches - from grid of patch to batch of patch 
get_patches - create patches from image
infer_model - run inference with pre post processing
fill_z - extract features, put it in z matrix, sub_batch - 64 patches are heavy for gpu, process & merge mini batches 
run_patch_model - to get results from two models at once
count_parameters
"""

def combine_global(batch_z_matrix, global_feat_resized, method='add'):
    if method=='cat':
        batch_z_matrix = torch.cat([batch_z_matrix, global_feat_resized], dim=1)
    elif method=='add':
        batch_z_matrix = batch_z_matrix + global_feat_resized
    else: # default is add
        batch_z_matrix = batch_z_matrix + global_feat_resized

    return batch_z_matrix

def sample_k_patches(patches, k, patch_side=4):
    k_patches = []
    positions = []
    currentPositions = {}
    idx = 0
    while idx < k:
        x, y = np.random.randint(0, patch_side, size=2)
        if (x, y) in currentPositions: continue

        currentPositions[(x, y)] = None
        k_patches.append(patches[x][y].unsqueeze(0)) 
        positions.append((x, y))
        idx = idx+1
    k_patches = torch.concat(k_patches, axis=0)
    return k_patches, positions

def merge_patches(mats, patch_side):
    """
    2, 4, 4, (3, 2048, 2048) -> (2, 3, 2048, 2048)
    """
    reduce_batch = False # for backward compatability
    if len(mats.shape)==6: # [batch, patch m, patch n, channel, height, width]
        pass
    elif len(mats.shape)==5: # [patch m, patch n, channel, height, width]
        # reduce_batch = True
        mats = [mats] # batch dimension added

    merged_batch = []
    for mat in mats:
        final_mat = []
        # joining columns first
        for row in range(patch_side):
            # joining columns first
            in_mat = torch.concat([col for col in mat[row]], axis=2) # [64, 128, 128+128+...]
            final_mat.append(in_mat) # [64, 128+128+..., 512]

        merged_op = torch.concat(final_mat, axis=1) # [64, 512, 512]
        # print("--size of the image", merged_op.shape)
        merged_batch.append(merged_op.unsqueeze(0))

    merged_img = torch.cat(merged_batch, dim=0)
    if reduce_batch:
        merged_img = merged_img.squeeze()

    return merged_img

def batch_patches(patches, m, n, patch_size, channel=3):
    patches = torch.reshape(
        patches, (m * n, channel, patch_size, patch_size)
    )
    return patches

def get_patches(image, m, n, patch_size, channel=3):
    # Create matrix of image 
    # (3, 2048, 2048) -> 4, 4, (3, 2048, 2048)
    img_c, img_h, img_w = image.shape

    # ensure image size is divisible properly by patch size
    assert img_h%patch_size == 0, "E: Image height not div by patch - resize it"
    assert img_w%patch_size == 0, "E: Image width not div by patch - resize it"

    border_pad = 0
    patches = torch.empty(
        ( m, n, channel, patch_size+border_pad, patch_size+border_pad), 
        dtype=torch.float32)
    # print(patches.shape)

    i = 0
    for x in range(0, img_h, patch_size):
        j = 0
        for y in range(0, img_w, patch_size):
            patch_image = image[:, x:x+patch_size, y:y+patch_size].clone()
            patches[i][j] = torch.Tensor(patch_image) 
            j = j+1
        i += 1

    return patches

def infer_model(model, batch_patch, patch_size, pre_post_pad=False):
    border_pad = 64
    pad_transform = torchvision.transforms.Pad(padding=border_pad, fill=0, padding_mode='reflect')
    center_crop_transform = torchvision.transforms.CenterCrop(size=(patch_size, patch_size))

    if pre_post_pad: batch_patch = pad_transform(batch_patch) # PRE TRANSFORM 
    model_op = model(batch_patch)
    if pre_post_pad: model_op = center_crop_transform(model_op) # POST TRANSFORM

    # print(k, 'model_op:', model_op.shape, "input images:", batch_patch.shape, "patch len:", len(batch_patch))
    return model_op

def fill_z(model, inputs, patch_side, device='cuda', feature_size=64, per_batch = 20):
    """
    this works well, as expected
    """
    if feature_size==None: feature_size=64
        
    input_dim = inputs.shape[-1]
    patch_size = int(input_dim/patch_side)

    op = []
    img_patches = [] # list of patches
    img_patches_feats = [] # list of features of patches

    for i, i_x in enumerate(inputs): # READ SINGLE IMAGE (3, 2048, 2048)
        patches = get_patches(i_x, m=patch_side, n=patch_side, patch_size=patch_size) # 2048 # matrix [4,4,3,128]
        # print("Z Matrix, Patches::", patches.shape)
        batch_patch = batch_patches(patches, m=patch_side, n=patch_side, patch_size=patch_size).to(device) # [16,3,128]
        # batch_patch_op = model(batch_patch) # WITHOUT SUB BATCHING

        ################[ SUB BATCHING ]########################################
        # IF LARGE NUMBER OF PATCHES, THEN NEED TO SUB-BATCH IT
        interm_op_l = []
        for k in range(0, len(batch_patch), per_batch): # (4x4, bs 10) -> 16-> 10,6 -> 16  
            k_batch_patch = batch_patch[k:k+per_batch]
            model_op = model(k_batch_patch)['out']
            # model_op = model_op.cpu().requires_grad_(True)
            model_op = model_op.requires_grad_(True)


            # print("batch", k_batch_patch.shape) # [10, 3, 128, 128]
            # print("Model OP:", model_op.shape)  # [10, 64, 128, 128]
            # interm_op_l.append(model_op.detach().cpu().requires_grad_(True))
            interm_op_l.append(model_op)
        batch_patch_op = torch.cat(interm_op_l, axis=0)
        ########################################################################
        
        # print("Batch Patch Op:",batch_patch_op.shape)
        # CONVERTING LINEAR FEATURES, BACK TO MATRIX - # 16,(64,256,256) -> 4,4,(64,256,256)
        batch_patch_op_matrix = torch.reshape(
            batch_patch_op,
            (patch_side, patch_side, feature_size, patch_size, patch_size)) 

        img_patches.append(patches.unsqueeze(0))
        img_patches_feats.append(batch_patch_op_matrix.unsqueeze(0))

    return torch.concatenate(img_patches), torch.concatenate(img_patches_feats)

def run_patch_model(model1, model2, x, y=None, patch_side=4, device='cuda', feature_size=None, use_global_feat=False, method=None, ip_size=None, patch_size=None, verbose=False):
    loss_fn = DiceBCELoss()
    epoch_loss = 0.0

    x = x.to(device, dtype=torch.float32) # can be batched image
    img_b, img_c, img_h, img_w  = x.shape
    with torch.no_grad():
        b_img_patches, b_img_patches_feats = fill_z(model1, x, patch_side, feature_size=feature_size) # batch of images
        batch_z_matrix = merge_patches(b_img_patches_feats, patch_side=patch_side)
        batch_z_matrix = batch_z_matrix.to(device)

        if use_global_feat:
            resizedImage = torchvision.transforms.Resize( size=(patch_size, patch_size))(x)
            global_feat = model1(resizedImage)['out']
            global_feat_resized = torchvision.transforms.Resize( size=ip_size)(global_feat)
            batch_z_matrix = combine_global(batch_z_matrix, global_feat_resized, method=method)
            
    y_pred = model2(batch_z_matrix)
    if verbose:
        print(f"Patch Side - {patch_side} - Model 1 Input: [{x.shape}] => Output: [{b_img_patches_feats.shape}] -> [{batch_z_matrix.shape}]")
        print(f"Model 2 Input: [{batch_z_matrix.shape}] => Output: [{y_pred.shape}]")

    if not (y is None):
        y = y.to(device, dtype=torch.float32) # can be batched image
        loss = loss_fn(y_pred, y) # already sigmoid is applied here
        epoch_loss += loss.item()
        return epoch_loss   

    y_pred = torch.sigmoid(y_pred)
    return y_pred, batch_z_matrix

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
