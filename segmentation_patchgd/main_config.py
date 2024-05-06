from src.utils import load_config

# "full", "down", "pgd", "pgd_global", "tiled"
config_dataset = {
    "aerial": {
        "train_data_root"   : "vinayv/data/dubai_dataset",
        "eval_data_path"    : "vinayv/data/dubai_dataset/test",
        "train_folder"      : "train",
        "valid_folder"      : "test",
        "ip_size"           : 1024, # for upscaling or downscaline
        "gt_size"           : 1024, # for comparing gt with pred results
        "down_size"         : 256, # in case downsampling is done
        "opt_patch_size"    : 128 # for optimized use cases
        # "patch_size":256, # in case patching is done
    },
    "drive": {
        "train_data_root"  : "vinayv/data/retina",
        "eval_data_path"   : "vinayv/data/retina/test",
        "train_folder"     : "train",
        "valid_folder"     : "test",
        "ip_size"          : 512,
        "gt_size"          : 512,
        "down_size"        : 128,
        "opt_patch_size"   : 128
        # "patch_size"     : 128,
    }
}

config_model_type = { "full", "down", "pgd", "pgd_global", "tiled"}

class CFG:
    def __init__(self, config_path=None):
        # config_path = "config.yaml"
        config_file = load_config(config_path)
        self.model_type = None
        self.data_name = None
        self.global_feat = False
        self.global_method = None
        self.feature_size = 1

        self.exp_name          = config_file.exp_name
        self.model             = config_file.model
        self.model_type        = config_file.model_type
        self.data_name         = config_file.data
        self.patch_size        = config_file.patch_size

        self.model1_path       = config_file.model1_path
        self.model2_path       = config_file.model2_path

        self.batch_size        = config_file.batch_size
        self.val_interval      = config_file.val_interval
        self.num_epoch         = config_file.num_epoch

        self.update()

    
    def update(self):
        """
        to update newly generated values
        """

        self.train_data_root         = config_dataset[self.data_name]["train_data_root"]
        self.eval_data_path          = config_dataset[self.data_name]["eval_data_path"]
        self.train_folder            = config_dataset[self.data_name]["train_folder"]
        self.valid_folder            = config_dataset[self.data_name]["valid_folder"]
        self.ip_size                 = config_dataset[self.data_name]["ip_size"]
        self.gt_size                 = config_dataset[self.data_name]["gt_size"]
        self.down_size               = config_dataset[self.data_name]["down_size"]

        self.patch_side              = int(self.ip_size/self.patch_size)
        self.data_split_patch_side   = 1

        if self.model_type == "full":
            self.patch_side = 1
            self.ip_size = self.gt_size

        elif self.model_type == "down":
            self.patch_side = 1
            self.ip_size = self.down_size

        elif self.model_type == "tiled":
            self.data_split_patch_side = self.patch_side
            self.batch_size = self.batch_size * self.data_split_patch_side

        elif self.model_type == "pgd":
            self.use_patchgd     = True
            self.n_inner_iter    = 4
            self.n_acc_steps     = 2
            self.k_patches       = 4
            self.feature_size    = 8

        elif self.model_type == "pgd_global":
            self.use_patchgd     = True
            self.n_inner_iter    = 4
            self.n_acc_steps     = 2
            self.k_patches       = 4
            self.feature_size    = 8
        
            self.global_feat     = True
            self.global_method   = "cat"

        else:
            print("Error - Select Correct Model Type")
            exit()

        self.patch_name = f"{self.model}_{self.data_name}_{self.model_type}_ps_{self.patch_side}_{self.ip_size}"
        self.op_dir = f"../OUTPUT_v8/files_{self.patch_name}_{self.exp_name}"


if __name__=="__main__":
    config = CFG("config.yaml")
    print(config.model_type)
    print(config.global_feat)
    config.model_type="pgd_global"
    print(config.model_type)
    print(config.global_feat)
    config.update()
    print(config.model_type)
    print(config.global_feat)
    