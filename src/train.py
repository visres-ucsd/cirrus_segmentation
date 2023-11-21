from keras_segmentation.models.unet import vgg_unet

model = vgg_unet(n_classes=2,  input_height=416, input_width=608)

model.train(
    train_images =  "ILM_cirrus_train/training/",
    train_annotations = "ILM_cirrus_train/truth/",
    checkpoints_path = "logs/ILM_cirrus_train" , epochs=25
)
