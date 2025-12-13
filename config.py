class Config:
    batch_size = 1024
    latent_dim = 128
    target_frames = 128
    model_type = 'ae'  # "vae", "ae", "byol", "contrastive"
    contrastive_aug = True
    lr = 1e-3
    epochs = 71
    use_augment = True
    n_clusters = 20