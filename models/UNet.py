from segmentation_models_pytorch import Unet

num_classes = 7

aux_params = {
    "classes": num_classes,
    "pooling": "max",
    "activation": "sigmoid"  # Each mask can have multiple classes
}

unet_model = Unet(
    encoder_name = "resnet34",
    classes = num_classes,
    in_channels = 1,       # Input image is grey
    dropout = 0.2,
    activation = "softmax", # Softmax because only once class per pixel
    aux_params = aux_params
)