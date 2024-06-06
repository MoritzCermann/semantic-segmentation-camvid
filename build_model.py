# Implementing a Semantic Segmentation model using the U-NET architecture

from tensorflow.keras import layers, Model, Input

activation = 'gelu'
# upconv_kernel_size = (3,3)
upconv_kernel_size = (2, 2)

def Conv2DBlock(input_tensor, filters, kernel_size, USE_BN = True):
    '''
    Build a block of 2 Conv2D layers for U-Net encoder & decoder
    Return block output
    '''
    x = layers.Conv2D(filters, kernel_size, padding='same')(input_tensor)
    if USE_BN:
        x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)

    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    if USE_BN:
        x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    return x


def build_unet(input_shape, num_classes, num_filters = 16, kernel_size = 3):
    '''
    Build a U-Net model
    Return model
    '''
    x = inputs = Input(input_shape)

    # Build encoder
    e1 = Conv2DBlock(x, num_filters * 1, kernel_size)
    p1 = layers.MaxPooling2D(2, padding='same')(e1)

    e2 = Conv2DBlock(p1, num_filters * 2, kernel_size)
    p2 = layers.MaxPooling2D(2, padding='same')(e2)

    e3 = Conv2DBlock(p2, num_filters * 4, kernel_size)
    p3 = layers.MaxPooling2D(2, padding='same')(e3)

    e4 = Conv2DBlock(p3, num_filters * 8, kernel_size)
    p4 = layers.MaxPooling2D(2, padding='same')(e4)

    latent = Conv2DBlock(p4, num_filters * 16, kernel_size)

    # Build decoder
    d4 = layers.Conv2DTranspose(num_filters * 8, upconv_kernel_size, 2, padding='same', activation=activation)(latent)
    d4 = layers.concatenate([d4, e4])
    d4 = Conv2DBlock(d4, num_filters * 8, kernel_size)

    d3 = layers.Conv2DTranspose(num_filters * 4, upconv_kernel_size, 2, padding='same', activation=activation)(d4)
    d3 = layers.concatenate([d3, e3])
    d3 = Conv2DBlock(d3, num_filters * 4, kernel_size)

    d2 = layers.Conv2DTranspose(num_filters * 2, upconv_kernel_size, 2, padding='same', activation=activation)(d3)
    d2 = layers.concatenate([d2, e2])
    d2 = Conv2DBlock(d2, num_filters * 4, kernel_size)

    d1 = layers.Conv2DTranspose(num_filters * 1, upconv_kernel_size, 2, padding='same', activation=activation)(d2)
    d1 = layers.concatenate([d1, e1])
    d1 = Conv2DBlock(d1, num_filters * 1, kernel_size)

    outputs = layers.Conv2D(num_classes, 1, padding="same", activation = "softmax")(d1)
    model = Model(inputs, outputs, name='U-Net')
    return model

