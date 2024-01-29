from architectures import ViT, PVT, SwinTransformer

#tiny version of all models

def load_model(model_name):
    if model_name == 'vit':
       
        model = ViT(img_dim=32,
            patch__dim=8, 
            embed_dim=49, 
            num_classes=10, 
            n_heads=3, 
            depth=2)

    elif model_name == 'pvt':

        model = PVT(img_dim=256,
            in_chans=3,
            patch_dim=4, 
            num_stages=4,
            embed_dims=[64, 128, 256, 512], 
            encoder_layers=[1, 1, 1, 1],
            reduction_ratio=[4, 2, 2, 1],
            n_heads=[1, 2, 4, 8],
            expansion_ratio=[6, 6, 4 ,4],
            num_classes=10)

    elif model_name == 'swin':

        model = SwinTransformer(in_chans=3, hidden_dim=96, 
                                layers=[2, 2, 6, 2], heads=[3, 6, 12, 24],
                                downscaling_factors=[4, 2, 2, 2], window_size=7, 
                                num_classes=10)

    else:
        raise ValueError(f"Modelo '{model_name}' no reconocido.")
    
    return model
