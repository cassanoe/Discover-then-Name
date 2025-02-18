

autoencoder_input_dim_dict = {'clip_RN50_out': 1024,
                              'clip_ViT-B16_out': 512,  
                              'clip_ViT-L14_out': 768, }

data_dir_root = '/scratch/cbm/dncbm/data'
save_dir_root = '/scratch/cbm/dncbm/SAE'
probe_cs_save_dir_root = '/scratch/cbm/dncbm/probe'
vocab_dir = '/scratch/cbm/dncbm/vocab'
analysis_dir = '/scratch/cbm/dncbm/analysis'



probe_dataset_root_dir_dict = {
    "places365": "/scratch/cbm/dncbm/data/activations_img/places365",
    "imagenet": "/shared/datasets/classification/imagenet",
    "cifar10": "/scratch/cbm/dncbm/data/activations_img/cifar10",
    "cifar100": "/scratch/cbm/dncbm/data/activations_img/cifar100",
}

probe_dataset_nclasses_dict = {"places365": 365,
                               'imagenet': 1000, "cifar10": 10, "cifar100": 100, }
