from keras_segmentation.models.unet import vgg_unet

model = vgg_unet(n_classes=2,  input_height=416, input_width=608)

model.train(
    train_images =  "cirrus/RNFL_cirrus/training/",
    train_annotations = "cirrus/RNFL_cirrus/truth/",
    checkpoints_path = "logs/RNFL/RNFL" , epochs=25
)
