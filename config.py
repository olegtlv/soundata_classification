class Config:
    batch_size = 1024
    latent_dim = 128
    target_frames = 128
    model_type = 'contrastive'  # "vae", "ae", "byol", "contrastive"
    contrastive_aug = True
    lr = 1e-3
    epochs = 171
    use_augment = True
    pretrained=True
    n_clusters = 10
    mode = 'simclr'  # byol, "ae", simclr