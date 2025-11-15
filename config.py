class Config:
    batch_size = 256
    latent_dim = 128
    target_frames = 128
    model_type = "ae"  # "vae", "simclr", "byol"
    contrastive_aug = True
    lr = 2e-3
    epochs = 20
    use_augment = True