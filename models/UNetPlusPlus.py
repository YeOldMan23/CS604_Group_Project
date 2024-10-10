from segmentation_models_pytorch import UnetPlusPlus

num_classes = 9

aux_params = {
    "classes": num_classes,
    "pooling": "max",
    "activation": "sigmoid" # Each mask can have multiple classes
}

unet_plus_plus_model = UnetPlusPlus(
    encoder_name = "resnet34",
    classes = num_classes,
    in_channels = 1,       # Input image is grey
    activation = "softmax", # Softmax because only once class per pixel
    aux_params = aux_params
)