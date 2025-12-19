class Config:
    batch_size = 1024
    latent_dim = 128
    target_frames = 128
    model_type = 'byol'  # "vae", "ae", "byol", "contrastive"
    contrastive_aug = True
    lr = 2e-3
    epochs = 71
    use_augment = True
    pretrained=True
    n_clusters = 10