import os
import torch
import random
import time
import json 
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torchvision

from glob import glob
from tqdm import tqdm 
from torchviz import make_dot

from operator import add
# import sys
# sys.path.append('codes/unetv4/')
from src.patchgd.patchgd_utils import run_patch_model, get_patches, batch_patches
# from patchgd.patchgd_module import Identity, patch_process, z_block_v4 as z_block, final_model, patch_process_func
from src.metrics import get_extended_stats, analyse_extended_scores, get_metrics, calculate_metrics, analyse_scores

def plot_graph(train_losses, valid_losses, path):
    plt.plot(train_losses, color='orange')
    plt.plot(valid_losses, color='blue')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(path)

def seeding(seed):
    """ Seeding the randomness. """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True 

def create_dir(path, safety=True):
    """ Create a directory. """
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        print(f"!!! Directory Already Exists - Change Experiment Name !!! [{path}]")
        # TODO: workaround to do testing without extra flags added
        ip = input("Replace 'y' or Enter: ")
        
        if not ip=='y':
            if path.split('_')[-1]=='test':
                safety=False
            if safety: exit()
        else:
            print(f"!!! Replacing the Files !!!  [{path}]")

            
def epoch_time(start_time, end_time):
    """ Calculate the time taken """
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def train(model, loader, optimizer, loss_fn, device):
    epoch_loss = 0.0

    model.train()
    for x, y in tqdm(loader):
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    # add gradient accumulation step also
    """
    # but we have created more batches, so it should work
    # only when batch size is 1 - this need to be added
    
    loss.backward(retain_graph=True)
    if (j+1) % constants.accumulationSteps == 0:
        optimizer.step() 
    """
    epoch_loss = epoch_loss/len(loader)
    return epoch_loss

def evaluate(model, loader, loss_fn, device):
    epoch_loss = 0.0
    print("Validation Started")

    model.eval()
    metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0]
    count = 0

    with torch.no_grad():
        for x, y in tqdm(loader):
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()

            for iy_pred, iy in zip(y_pred, y):
                count+=1
                iy_pred = iy_pred.unsqueeze(dim=0)
                iy = iy.unsqueeze(dim=0)
                pred_y = torch.sigmoid(iy_pred)
                score = calculate_metrics(iy, pred_y)
                metrics_score = list(map(add, metrics_score, score))

        jaccard = metrics_score[0]/count
        f1 = metrics_score[1]/count
        recall = metrics_score[2]/count
        precision = metrics_score[3]/count
        acc = metrics_score[4]/count

        epoch_loss = epoch_loss/len(loader)
        print(f"Jaccard: {jaccard:1.4f} - F1: {f1:1.4f} - Recall: {recall:1.4f} - Precision: {precision:1.4f} - Acc: {acc:1.4f} - Count: {count}")
    return epoch_loss


def load_checkpoint(path, model):
    state_dict = torch.load(path)
    state_dict = {key.replace('module.',''):state_dict[key] for key in state_dict.keys()}
    model.load_state_dict(state_dict)
    model.eval()
    return model



def get_stats(x):
    try:
        x = np.array(x)
    except:
        x = x.detach().cpu().numpy()
    print("Stats:", np.min(x), np.max(x), np.unique(x))


def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1)    ## (512, 512, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1)  ## (512, 512, 3)
    return mask

def xx_process_mask(mask):
    H = 512
    W = 512
    size = (W, H)
    print("ip:", mask.shape, end=' ')
    mask = cv2.resize(mask, size)
    print("op:", mask.shape)
    mask = mask/255.0
    return mask

def preprocess_image(path, op_size, ip_size):
    """ Reading image """
    image = cv2.imread(path, cv2.IMREAD_COLOR) ## (512, 512, 3)
    image = cv2.resize(image, op_size)
    x_image = cv2.resize(image, ip_size)

    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # print(image.shape)
    # print(np.unique(image))
    
    # b_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    x = np.transpose(x_image, (2, 0, 1))      ## (3, 512, 512)
    x = x/255.0
    x = np.expand_dims(x, axis=0)           ## (1, 3, 512, 512)
    x = x.astype(np.float32)
    x = torch.from_numpy(x)
    return image, x

def preprocess_mask(path, size):
    """ Reading mask 
    input of mask is in range 0-255 
    gray pixels are also present in it - which are middle ones

    mask should only be binary - nothing in midrange
    """
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  ## (512, 512)
    mask = cv2.resize(mask, size)
    mask = mask/255.0
    mask = mask > 0.5
    # y = (y-1)*-1 # invert mask
    # inv_mask = y
    y = np.expand_dims(mask, axis=0)            ## (1, 512, 512) channel add
    y = np.expand_dims(y, axis=0)               ## (1, 1, 512, 512) batch add
    y = y.astype(np.float32)
    # y = torch.from_numpy(y)
    # print("preprocess mask:",np.unique(y))
    return mask, y

def postprocess_pred(pred_y, thresh=0):
    pred_y = pred_y[0] ## (1, 512, 512)
    pred_y = np.squeeze(pred_y, axis=0)     ## (512, 512)
    # print(pred_y.shape)
    # pred_y = pred_y > thresh
    pred_y = np.array(pred_y* 255, dtype=np.uint8)
    return pred_y

def add_text(img, text, size=None):
    # print(text, img.shape, img.max(), img.min())
    # print(np.unique(img))
    img = np.array(img, dtype='uint8')
    img = cv2.resize(img, size)
    # font = cv2.FONT_HERSHEY_SIMPLEX
    # img = cv2.putText(img, text, (10,25), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    return img

def write_image(image, y_gt, y_pred, op_path, zmatrix, size):
    y_pred = postprocess_pred(y_pred, thresh=0.5)
    ori_mask = mask_parse(y_gt*255)
    # print("write mask:",np.unique(y_gt))
    # print("write original mask:",np.unique(ori_mask))
    # print("write pred mask:",np.unique(ori_mask))
    # inv_mask = mask_parse(inv_mask)
    zmatrix = mask_parse(zmatrix)

    # print(y_pred)
    thres_y_pred = (y_pred>127)*255

    # print(thres_y_pred)
    y_pred = mask_parse(y_pred)
    thres_y_pred = mask_parse(thres_y_pred)

    line = np.ones((size[1], 10, 3)) * 128

    image = add_text(image, text="ip image", size=size)
    ori_mask = add_text(ori_mask, text="gt mask", size=size)
    y_pred = add_text(y_pred, text="pred mask", size=size)
    thres_y_pred = add_text(thres_y_pred, text="thresh mask", size=size)
    zmatrix = add_text(zmatrix, text="z matrix", size=size)

    cat_images = np.concatenate(
        [image, line, ori_mask, line, y_pred, line, thres_y_pred, line, zmatrix], axis=1
    )
    cv2.imwrite(op_path, cat_images)




def plot_z_channels(c_split):
    import matplotlib.pyplot as plt
    # [cv2.imwrite(f"z_mat_{i}.jpg", c*255) for i, c in enumerate(c_split)]
    # total_img = 64
    # cols = 8
    # rows = int(total_img/cols)
    # fig, axs = plt.subplots(rows,cols)
    # for i, file in enumerate(files):
    #     y, x = int(i%cols), int(i/cols)
    #     # print(x,y)
    #     axs[y, x].imshow(Image.open(file))
    #     axs[y, x].axis(False)
    #     axs[y, x].set_title(f"C:{i}", fontsize=5)
        
    fig, axs = plt.subplots(8, 8)
    for i, channel in enumerate(c_split):
        axs[int(i/8), i%8].imshow(channel*255, cmap='gray')
        axs[int(i/8), i%8].axis(False)
        axs[int(i/8), i%8].set_title(f"C:{i}", fontsize=5)

    fig.tight_layout()
    fig.savefig("final_z.png")
    
def get_zmetrics_mean(z_matrix):
    z_channels = z_matrix[0]
    # print(z_matrix.shape, z_channels.shape)
    c_split = [i.cpu().detach().numpy() for i in z_channels]
    # plot_z_channels(c_split)
    mean_img = np.mean(c_split, axis=0)
    mean_img = 1/(1 + np.exp(-mean_img)) 
    # print(mean_img)
    # cv2.imwrite('visualize_z.jpg',mean_img)
    # exit()
    return mean_img

def run_test_on_model(model1, model2, result_folder, patch_side, sample=None, device='cuda', feature_size=None, ip_size=None, gt_size=None, data_path=None, use_global_feat=False, patch_size=None, method=None):

    if type(ip_size)==int:
        ip_size = (ip_size, ip_size)

    if type(gt_size)==int:
        gt_size = (gt_size, gt_size)

    model1.to(device).eval()
    model2.to(device).eval()

    test_x = sorted(glob(f"{data_path}/image/*"))
    test_y = sorted(glob(f"{data_path}/mask/*"))

    
    if not sample is None:
        test_x = test_x[:sample]  
        test_y = test_y[:sample]

    # H, W = 64, 64
    H, W = 512, 512
    # H, W = 1024, 1024
    op_size = (W, H)
    time_taken = []
    scores = []
    extended_scores = []
    create_dir(result_folder)

    # sigmoid = torch.sigmoid()
    # exit()
    print(f"==--> GT Size: {gt_size}, Model Input Shape: {ip_size}")

    start = True
    for i, (x_pth, y_pth) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        """ Extract the name """

        name = x_pth.split("/")[-1].split(".")[0]
        image, x = preprocess_image(x_pth, gt_size, ip_size)
        x = x.to(device)
        gt_mask, y_gt = preprocess_mask(y_pth, gt_size) # already 0, 1 gt

        start_time = time.time()
        y_pred, z_matrix = run_patch_model(model1, model2, x, patch_side=patch_side, device=device, feature_size=feature_size, use_global_feat=use_global_feat, method=method, ip_size=ip_size, patch_size=patch_size, verbose=start)
        total_time = time.time() - start_time

        if start: 
            start=False
            
        mean_img = get_zmetrics_mean(z_matrix)
        if gt_size==None:
            gt_size = y_gt.shape[2:]

        y_pred = torchvision.transforms.Resize(size=gt_size)(y_pred)
        y_pred = y_pred.detach().cpu().numpy()
        # print(y_pred.shape)
        y_pred = np.array(y_pred>0.5, dtype=np.float32)


        # one image at a time
        score = get_metrics(y_pred, y_gt, gt_size) # 0 and 1 are going here
        # print("=== Score", score)
        extended_score = get_extended_stats(output=y_pred, target=y_gt, get_list=True)
        extended_scores.append(extended_score)

        time_taken.append(total_time)
        scores.append(score)

        # print("Predictions :", x.shape, y_pred.shape)
        op_path = f"{result_folder}/{name}.png"
        write_image(image, gt_mask, y_pred, op_path, mean_img*255, op_size)

    analyse_scores(scores)
    analyse_extended_scores(extended_scores)
    fps = 1/np.mean(time_taken)
    print("FPS: ", fps)

def generate_small_patches(x, y, patch_side=4, concat=True):
    new_x, new_y = [], []
    input_dim = x.shape[-1]
    patch_size = int(input_dim/patch_side)
    for i_x, i_y in zip(x, y):
        # print(i_x.shape, i_y.shape)
        x_patches = get_patches(i_x, m=patch_side, n=patch_side, patch_size=patch_size) # 2048
        x_batch_patch = batch_patches(x_patches, m=patch_side, n=patch_side, patch_size=patch_size)
        new_x.append(x_batch_patch)

        y_patches = get_patches(i_y, m=patch_side, n=patch_side, patch_size=patch_size, channel=1) # 2048
        y_batch_patch = batch_patches(y_patches, m=patch_side, n=patch_side, patch_size=patch_size, channel=1)
        new_y.append(y_batch_patch)

    if not concat:
        return new_x, new_y

    new_x = torch.cat(new_x, dim=0)
    new_y = torch.cat(new_y, dim=0)
    # print(new_x.shape, new_y.shape)
    return new_x, new_y


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return model


def load_config(config_path):
    import yaml
    class dotdict(dict):
        """dot.notation access to dictionary attributes"""
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__

    config = yaml.load(open(config_path, 'r'), yaml.Loader)
    # print(config)
    return (dotdict(config))

def get_computational_graph(var, models):
    # models = [model1, model2]
    ## COMPUTATIONAL GRAPH CHECK
    def get_dict_params(models):
        params = {}
        for i, m in enumerate(models):
            p_dict = dict(m.named_parameters())
            p_dict = {f"m{i+1}-"+k:p_dict[k] for k in p_dict.keys()}
            # print(p_dict)
            params.update(p_dict)
        return params

    params = get_dict_params(models)
    print(params)
    make_dot(var, params=params).view()
    exit()


if __name__=='__main__':
    mask_path = "image/0.png"
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  ## (512, 512)
    mask1, mask2 = mask, mask
    mask2 = mask2[:, ::-1]
    # mask1 = np.ones((700,700))*255
    # mask2 = np.zeros((700,700))*255