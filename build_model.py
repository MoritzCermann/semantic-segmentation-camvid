# Implementing a Semantic Segmentation model using the U-NET architecture

from tensorflow.keras import layers, Model, Input


def encoder_block(x, num_filters, kernel_size):
    # conv 3x3, ReLu
    x = layers.Conv2D(num_filters, kernel_size, padding='same')(x)
    x = layers.Activation('relu')(x)
    # conv 3x3, ReLu
    x = layers.Conv2D(num_filters, kernel_size, padding='same')(x)
    x = layers.Activation('relu')(x)
    
    skip_connection = x
    # max pool 2x2
    x = layers.MaxPooling2D(2)(x)
    return x, skip_connection

def bottleneck(x, num_filters, kernel_size):
    # conv 3x3, ReLu
    x = layers.Conv2D(num_filters, kernel_size, padding='same')(x)
    x = layers.Activation('relu')(x)
    # conv 3x3, ReLu
    x = layers.Conv2D(num_filters, kernel_size, padding='same')(x)
    x = layers.Activation('relu')(x)
    return x

def decoder_block(x, skip_layer, num_filters, kernel_size):
    # up-conv 2x2
    x = layers.Conv2DTranspose(num_filters, (2, 2), strides=2, padding='same')(x)
    # concatenate with copy
    x = layers.concatenate([x, skip_layer], axis=-1)
    # conv 3x3, ReLu
    x = layers.Conv2D(num_filters, kernel_size, padding='same')(x)
    x = layers.Activation('relu')(x)
    # conv 3x3, ReLu
    x = layers.Conv2D(num_filters, kernel_size, padding='same')(x)
    x = layers.Activation('relu')(x)    
    return x
    
    
def build_unet(input_shape, num_classes, num_filters=16, kernel_size=3):
    '''
    Build a U-Net model
    Return model
    '''
    inputs = Input(input_shape)

    # Encoder - Contracting path - Downsampling
    e1, skip_e1 = encoder_block(inputs, num_filters, kernel_size)
    e2, skip_e2 = encoder_block(e1, num_filters * 2, kernel_size)
    e3, skip_e3 = encoder_block(e2, num_filters * 4, kernel_size)
    e4, skip_e4 = encoder_block(e3, num_filters * 8, kernel_size)

    # Bottleneck - Lowest layer
    lowest_layer = bottleneck(e4, num_filters * 16, kernel_size)

    # Decoder - Expanding Path - Upsampling
    d4 = decoder_block(lowest_layer, skip_e4, num_filters * 8, kernel_size)
    d3 = decoder_block(d4, skip_e3, num_filters * 4, kernel_size)
    d2 = decoder_block(d3, skip_e2, num_filters * 2, kernel_size)
    d1 = decoder_block(d2, skip_e1, num_filters, kernel_size)

    # conv 1x1
    outputs = layers.Conv2D(num_classes, 1, padding="same", activation="softmax")(d1)
    model = Model(inputs, outputs, name='U-Net')
    return model
