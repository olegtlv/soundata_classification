class Config:
    batch_size = 4096
    latent_dim = 128
    target_frames = 128
    model_type = "contrastive"  # "vae", "simclr", "byol"
    contrastive_aug = True
    lr = 2e-3
    epochs = 50
    use_augment = True
    n_clusters = 20