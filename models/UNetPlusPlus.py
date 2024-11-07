from segmentation_models_pytorch import UnetPlusPlus

num_classes = 9 # Exclude BG, combine left and right kidney into 1

aux_params = {
    "pooling": "max",
    "classes": num_classes,
    "pooling": "max",
    "dropout" : 0.2,
    "activation" : "sigmoid" # Each mask can have multiple classes
}

unet_plus_plus_model = UnetPlusPlus(
    encoder_name = "resnet34",
    classes = num_classes,
    in_channels = 1,       # Input image is grey
    activation = None, # Follow Along
    # decoder_attention_type = "scse",
    aux_params = aux_params
)