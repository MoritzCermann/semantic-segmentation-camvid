# Implementing a Semantic Segmentation model using the U-NET architecture

from tensorflow.keras import layers, Model, Input

activation = 'relu'
# upconv_kernel_size = (3,3)


def build_unet(input_shape, num_classes, num_filters=16, kernel_size=3):
    '''
    Build a U-Net model
    Return model
    '''
    inputs = Input(input_shape)

    # Transpose the input to (Channels, Height, Width)
    #x = layers.Lambda(lambda x: tf.transpose(x, (0, 3, 1, 2)))(inputs)

    # Encoder
    x = layers.Conv2D(num_filters, kernel_size, padding='same')(inputs)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(num_filters, kernel_size, padding='same')(x)
    e1 = layers.Activation('relu')(x)
    p1 = layers.MaxPooling2D(2)(e1)

    x = layers.Conv2D(num_filters * 2, kernel_size, padding='same')(p1)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(num_filters * 2, kernel_size, padding='same')(x)
    e2 = layers.Activation('relu')(x)
    p2 = layers.MaxPooling2D(2)(e2)

    x = layers.Conv2D(num_filters * 4, kernel_size, padding='same')(p2)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(num_filters * 4, kernel_size, padding='same')(x)
    e3 = layers.Activation('relu')(x)
    p3 = layers.MaxPooling2D(2)(e3)

    x = layers.Conv2D(num_filters * 8, kernel_size, padding='same')(p3)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(num_filters * 8, kernel_size, padding='same')(x)
    e4 = layers.Activation('relu')(x)
    p4 = layers.MaxPooling2D(2)(e4)

    # Latent space
    x = layers.Conv2D(num_filters * 16, kernel_size, padding='same')(p4)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(num_filters * 16, kernel_size, padding='same')(x)
    latent = layers.Activation('relu')(x)

    # Decoder
    x = layers.Conv2DTranspose(num_filters * 8, 2, strides=2, padding='same')(latent)
    x = layers.concatenate([x, e4], axis=1)
    x = layers.Conv2D(num_filters * 8, kernel_size, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(num_filters * 8, kernel_size, padding='same')(x)
    d4 = layers.Activation('relu')(x)

    x = layers.Conv2DTranspose(num_filters * 4, 2, strides=2, padding='same')(d4)
    x = layers.concatenate([x, e3], axis=1)
    x = layers.Conv2D(num_filters * 4, kernel_size, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(num_filters * 4, kernel_size, padding='same')(x)
    d3 = layers.Activation('relu')(x)

    x = layers.Conv2DTranspose(num_filters * 2, 2, strides=2, padding='same')(d3)
    x = layers.concatenate([x, e2], axis=1)
    x = layers.Conv2D(num_filters * 2, kernel_size, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(num_filters * 2, kernel_size, padding='same')(x)
    d2 = layers.Activation('relu')(x)

    x = layers.Conv2DTranspose(num_filters, 2, strides=2, padding='same')(d2)
    x = layers.concatenate([x, e1], axis=1)
    x = layers.Conv2D(num_filters, kernel_size, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(num_filters, kernel_size, padding='same')(x)
    d1 = layers.Activation('relu')(x)

    # Transpose the output back to (Height, Width, Channels)
    #x = layers.Lambda(lambda x: tf.transpose(x, (0, 2, 3, 1)))(d1)

    outputs = layers.Conv2D(num_classes, 1, padding="same", activation="softmax")(d1)
    model = Model(inputs, outputs, name='U-Net')
    return model

