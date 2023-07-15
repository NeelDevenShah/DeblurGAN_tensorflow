import tensorflow as tf
import keras.backend as k
from keras.applications.vgg16 import VGG16
from keras.models import Model
import numpy as np

# Note the image_shape must be multiple of patch_shape
image_shape = (256, 256, 3)


def l1_loss(y_true, y_pred):
    return k.mean(k.abs(y_pred - y_true))


vgg = VGG16(include_top=False, weights='imagenet', input_shape=image_shape)
# vgg_variables = vgg.trainable_variables
# loss_model = Model(vgg_variables)

loss_model = Model(
    inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
loss_model.trainable = False


@tf.function
def perceptual_loss(y_true, y_pred):
    features1 = loss_model(y_true)
    features2 = loss_model(y_pred)
    loss = tf.reduce_mean(tf.square(features1 - features2))
    return loss


def perceptual_loss_100(y_true, y_pred):
    return 100 * perceptual_loss(y_true, y_pred)


def wasserstein_loss(y_true, y_pred):
    return k.mean(y_true*y_pred)


def gradient_penalty_loss(self, y_ture, y_pred, averaged_samples):
    gradients = k.gradients(y_pred, averaged_samples)[0]
    gradients_sqr = k.square(gradients)
    gradients_sqr_sum = k.sum(
        gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
    gradient_l2_norm = k.sqrt(gradients_sqr_sum)
    gradient_penalty = k.square(1 - gradient_l2_norm)
    return k.mean(gradient_penalty)
