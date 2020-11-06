#%% Imports
import tensorflow as tf
import tensorflow.keras.layers as kl
from tensorflow.keras.regularizers import l2

#%% Global Tensorflow settings
devices = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(devices[0], True)
tf.keras.mixed_precision.experimental.set_policy(
    tf.keras.mixed_precision.experimental.Policy("mixed_float16")
)
#%% Define builder functions
# For network architecture see https://arxiv.org/ftp/arxiv/papers/1612/1612.05360.pdf
def resnet_layer(
    num_filters,
    kernel_size=3,
    strides=1,
    activation=kl.LeakyReLU(dtype=tf.float16),
    batchnorm=True,
):
    def resnet_layer_gen(inputs):
        conv = kl.Conv2D(
            num_filters,
            kernel_size=kernel_size,
            strides=strides,
            padding="same",
            kernel_initializer="he_normal",
            kernel_regularizer=l2(1e-3),
            bias_initializer="he_normal",
            bias_regularizer=None,
            activation=None,
        )
        x = conv(inputs)
        if activation:
            x = activation(x)
        if batchnorm:
            x = kl.BatchNormalization()(x)

        return x

    return resnet_layer_gen


def residual_block(num_filters, num_layers, kernel_size=3):
    def residual_block_gen(inputs):
        inputs = resnet_layer(num_filters, kernel_size)(inputs)
        x = inputs
        for i in range(num_layers):
            x = resnet_layer(num_filters, kernel_size)(x)
        x = kl.add([x, inputs])
        x = resnet_layer(num_filters, kernel_size)(x)
        return x

    return residual_block_gen


def build_segmentation_model(
    input_shape, blocks_depth, num_layers, initial_filters, num_classes
):
    inputs = kl.Input(shape=input_shape)
    x = resnet_layer(6, kernel_size=3)(inputs)
    num_filters = initial_filters
    residuals = []

    for i in range(blocks_depth):
        x = residual_block(num_filters, num_layers)(x)
        residuals.append(x)
        x = kl.MaxPooling2D(2)(x)
        num_filters *= 2

    # Bridge block
    x = residual_block(num_filters, num_layers)(x)
    for i in reversed(range(blocks_depth)):
        num_filters //= 2
        x = kl.Conv2DTranspose(num_filters, kernel_size=3, strides=2, padding="same")(x)
        x = residual_block(num_filters, num_layers)(x)
        x = kl.add([x, residuals[i]])

    x = resnet_layer(num_filters, kernel_size=3)(x)
    output = resnet_layer(
        num_classes, kernel_size=1, activation=kl.Activation("sigmoid"), batchnorm=False
    )(x)
    model = tf.keras.Model(inputs=inputs, outputs=output)
    return model


#%% Build model
if __name__ == "__main__":
    model = build_segmentation_model((240, 240, 3), 4, 3, 16)
