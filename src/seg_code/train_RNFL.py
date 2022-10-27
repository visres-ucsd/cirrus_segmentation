from keras_segmentation.models.unet import vgg_unet
from keras.callbacks import ModelCheckpoint

model = vgg_unet(n_classes=2,  input_height=416, input_width=608)
callbacks = [
    ModelCheckpoint(
        filepath="logs/RNFL/RNFL{epoch:05d}_{val_loss:.4f}_{val_accuracy:.4f}",
        save_weights_only=True,
        verbose=True
    )
]

model.train(
    checkpoints_path="logs/RNFL/RNFL",
    train_images="cirrus/RNFL_cirrus/training",
    train_annotations="cirrus/RNFL_cirrus/truth",
    validate=True,
    val_images="cirrus/RNFL_cirrus/val_training",
    val_annotations="cirrus/RNFL_cirrus/val_truth",
    batch_size=20,
    val_batch_size=20,
    epochs=20,
    callbacks=callbacks
)