import datetime
import math

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import (
    LearningRateScheduler,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)

import augmentation
import data_generator
import model_builder
from accuracy import iou_coef

EPOCHS = 200
MODEL_SAVE_PATH = ".\\saved_models\\fusionNet"


def dice_coef(smooth):
    def dice_coef(y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2.0 * intersection + smooth) / (
            K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + smooth
        )

    return dice_coef


def tversky_loss(y_true, y_pred):
    alpha = 0.1
    beta = 0.9

    ones = K.ones_like(y_true)
    p0 = y_pred  # proba that voxels are class i
    p1 = ones - y_pred  # proba that voxels are not class i
    g0 = y_true
    g1 = ones - y_true

    num = K.sum(p0 * g0, (0, 1, 2, 3))
    den = num + alpha * K.sum(p0 * g1, (0, 1, 2)) + beta * K.sum(p1 * g0, (0, 1, 2))

    T = K.sum(num / den)  # when summing over classes, T has dynamic range [0 Ncl]

    Ncl = K.cast(K.shape(y_true)[-1], "float32")
    return Ncl - T


# https://arxiv.org/abs/2006.14822
def focal_loss(alpha, gamma):
    def focal_loss_func(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon())
        logits = tf.math.log(y_pred / (1 - y_pred))

        weight_a = alpha * (1 - y_pred) ** gamma * y_true
        weight_b = (1 - alpha) * y_pred ** gamma * (1 - y_true)

        loss = (tf.math.log1p(tf.exp(-tf.abs(logits))) + tf.nn.relu(-logits)) * (
            weight_a + weight_b
        ) + logits * weight_b
        return loss

    return focal_loss_func


def lr_schedule(epoch):
    initial_lrate = 1e-7
    drop = 0.7
    epochs_drop = 20.0
    lr = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))

    return lr


def lr_schedule_manual(epoch):
    if epoch > 5:
        return 1e-6
    elif epoch > 2:
        return 3e-6
    else:
        return 5e-6


def train_model():
    generator = data_generator.SegmentationSequence(
        ".\\data\\train\\images",
        ".\\data\\train\\masks",
        augmenter=augmentation.augment,
        batch_size=4,
        output_img_size=(240, 240),
        steps_per_epoch=1000,
    )
    model = model_builder.build_segmentation_model((240, 240, 3), 4, 4, 20, 16)
    opt = tf.keras.optimizers.Adam(lr=lr_schedule(0), clipnorm=1.0)
    model.compile(
        optimizer=opt,
        loss=tversky_loss,
        metrics=[dice_coef(1.0)],
    )

    tb = TensorBoard(
        log_dir=".\\logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
        histogram_freq=1,
    )
    # lr_reducer = ReduceLROnPlateau(
    #     factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-8
    # )
    lr_scheduler = LearningRateScheduler(lr_schedule)
    checkpoints = ModelCheckpoint(
        ".\\ckpts",
        monitor="loss",
        verbose=0,
        save_weights_only=False,
        mode="auto",
        save_freq="epoch",
    )
    callbacks = [lr_scheduler, tb, checkpoints]
    history = model.fit(generator, epochs=EPOCHS, callbacks=callbacks)
    model.save(MODEL_SAVE_PATH)


if __name__ == "__main__":
    train_model()
