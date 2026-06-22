sam2_dict = {
    "b+" : {"config": "sam2_configs/sam2.1_hiera_b+.yaml", "weight": "../../../Pretrained-Models/SAM2/sam2.1_hiera_base_plus.pt"},
    "l" : {"config": "sam2_configs/sam2.1_hiera_l.yaml", "weight": "../../../Pretrained-Models/SAM2/sam2.1_hiera_large.pt"},
    "s" : {"config": "sam2_configs/sam2.1_hiera_s.yaml", "weight": "../../../Pretrained-Models/SAM2/sam2.1_hiera_small.pt"},
    "t" : {"config": "sam2_configs/sam2.1_hiera_t.yaml", "weight": "../../../Pretrained-Models/SAM2/sam2.1_hiera_tiny.pt"}
}

omni_dataset_dict = {
    "train": {"df_root": "../../Dataset/360VOTS/train.csv",
              "data_root": "../../Dataset/360VOTS/train",
              "weight_root": "../../Dataset/360VOTS/train_mask_weight"},
    "val": {"df_root": "../../Dataset/360VOTS/test.csv",
             "data_root": "../../Dataset/360VOTS/test"}
}

pano_dataset_dict = {
    "df_root": "../../Dataset/PanoVOS/train.csv",
    "data_root": "../../Dataset/PanoVOS/train",
    "weight_root": "../../Dataset/PanoVOS/train_mask_weight"
}

class Config:
    lr = 2e-4
    epoch_num = 80
    dataset = "360VOTS" # "360VOTS", "PanoVOS"
    correct_epoch_info = [(1, 2), (8, -1)]
    mem_start_epoch = 20
    backbone = "t"
    pretrain_path = None
    print(f"\nThe backbone is {backbone}.")
    sam2_config_file = sam2_dict[backbone]["config"]
    sam2_weight_file = sam2_dict[backbone]["weight"]
    video_size = 100
    test_sample_num = 6
    train_sample_num = 400
    group_frame_num = 8
    batch_size = 2
    log_freq = 10
    num_workers = 4
    eval_sample = 3
    epoch_save_freq = 10
    # model
    long_mem_size = 2
    long_mem_scale = 100
    mem_stride = 1
    save_path = f"result/weight/{dataset}/{backbone}"
