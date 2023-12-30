import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

def squeeze_excite_block(inputs, ratio=8):
    init = inputs
    channel_axis = -1
    filters = init.shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    x = Multiply()([init, se])
    return x

def stem_block(x, n_filter, strides):
    x_init = x

    ## Conv 1
    x = Conv2D(n_filter, (3, 3), padding="same", strides=strides)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(n_filter, (3, 3), padding="same")(x)

    ## Shortcut
    s = Conv2D(n_filter, (1, 1), padding="same", strides=strides)(x_init)
    s = BatchNormalization()(s)

    ## Add
    x = Add()([x, s])
    x = squeeze_excite_block(x)
    return x

def resnet_block(x, n_filter, strides=1):
    x_init = x

    ## Conv 1
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(n_filter, (3, 3), padding="same", strides=strides)(x)
    ## Conv 2
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(n_filter, (3, 3), padding="same", strides=1)(x)

    ## Shortcut
    s = Conv2D(n_filter, (1, 1), padding="same", strides=strides)(x_init)
    s = BatchNormalization()(s)

    ## Add
    x = Add()([x, s])
    x = squeeze_excite_block(x)
    return x

def aspp_block(x, num_filters, rate_scale=1):
    x1 = Conv2D(num_filters, (3, 3), dilation_rate=(6 * rate_scale, 6 * rate_scale), padding="SAME")(x)
    x1 = BatchNormalization()(x1)

    x2 = Conv2D(num_filters, (3, 3), dilation_rate=(12 * rate_scale, 12 * rate_scale), padding="SAME")(x)
    x2 = BatchNormalization()(x2)

    x3 = Conv2D(num_filters, (3, 3), dilation_rate=(18 * rate_scale, 18 * rate_scale), padding="SAME")(x)
    x3 = BatchNormalization()(x3)

    x4 = Conv2D(num_filters, (3, 3), padding="SAME")(x)
    x4 = BatchNormalization()(x4)

    y = Add()([x1, x2, x3, x4])
    y = Conv2D(num_filters, (1, 1), padding="SAME")(y)
    return y

def attetion_block(g, x):
    filters = x.shape[-1]

    g_conv = BatchNormalization()(g)
    g_conv = Activation("relu")(g_conv)
    g_conv = Conv2D(filters, (3, 3), padding="SAME")(g_conv)

    g_pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(g_conv)

    x_conv = BatchNormalization()(x)
    x_conv = Activation("relu")(x_conv)
    x_conv = Conv2D(filters, (3, 3), padding="SAME")(x_conv)

    gc_sum = Add()([g_pool, x_conv])

    gc_conv = BatchNormalization()(gc_sum)
    gc_conv = Activation("relu")(gc_conv)
    gc_conv = Conv2D(filters, (3, 3), padding="SAME")(gc_conv)

    gc_mul = Multiply()([gc_conv, x])
    return gc_mul

def positional_attention_module(x, n_filters):
    BS, H, W, C = x.shape

    query = Conv2D(n_filters, (1, 1))(x)
    query = BatchNormalization()(query)

    key = Conv2D(n_filters, (1, 1))(x)
    key = BatchNormalization()(key)

    query = tf.keras.layers.Reshape((H * W, -1))(query) 
    key = tf.keras.layers.Reshape((H * W, -1))(key)

    energy = tf.linalg.matmul(query, key, transpose_b=True)
    
    attention = Activation("softmax")(energy)

    value = Conv2D(n_filters, (1, 1))(x)
    value = BatchNormalization()(value)

    value = tf.keras.layers.Reshape((H * W, -1))(value) 
    out = tf.linalg.matmul(value, attention, transpose_a=True)

    out = tf.keras.layers.Reshape(x.shape[1:])(out)
    out = out + x

    return out

def channel_attention_module(x, n_filters):
    BS, H, W, C = x.shape

    query = tf.keras.layers.Reshape((-1, C))(x) 
    query = BatchNormalization()(query)

    key = tf.keras.layers.Reshape((-1, C))(x)
    key = BatchNormalization()(key)

    energy = tf.linalg.matmul(query, key, transpose_a=True)
    energy_2 = tf.math.reduce_max(energy, axis=1, keepdims=True)[0] - energy

    attention = Activation("softmax")(energy_2)

    value = Conv2D(n_filters, (1, 1))(x)
    value = BatchNormalization()(value)

    value = tf.keras.layers.Reshape((-1, C))(x)
    out = tf.linalg.matmul(attention, value, transpose_b=True)

    out = tf.keras.layers.Reshape(x.shape[1:])(out)
    out = out + x

    return out

def dual_attetion_block(x, n_filter):
    x_pam = Conv2D(n_filter, (3, 3), padding='same')(x)
    x_pam = BatchNormalization()(x_pam)
    x_pam_out = positional_attention_module(x_pam, n_filters=n_filter)
    x_pam = Conv2D(n_filter, (3, 3), padding='same')(x_pam_out)
    x_pam = BatchNormalization()(x_pam)
    x_pam = Dropout(0.1)(x_pam)
    x_pam = Activation("relu")(x_pam)

    x_cam = Conv2D(n_filter, (3, 3), padding='same')(x)
    x_cam = BatchNormalization()(x_cam)
    x_cam_out = channel_attention_module(x_cam, n_filters=n_filter)
    x_cam = Conv2D(n_filter, (3, 3), padding='same')(x_cam_out)
    x_cam = BatchNormalization()(x_cam)
    x_cam = Dropout(0.1)(x_cam)
    x_cam = Activation("relu")(x_cam)

    x_pam_cam = Concatenate(axis=3)([x_pam_out, x_cam_out])
    x_pam_cam = Dropout(0.1)(x_pam_cam)
    x_pam_cam = Activation("relu")(x_pam_cam)

    x_pam_cam = Concatenate(axis=3)([x_pam, x_cam, x_pam_cam])
    return x_pam_cam

class PAM_ResUnetPlusPlus:
    def __init__(self, input_size=64):
        self.input_size = input_size

    def build_model(self):
        n_filters = [32, 64, 128, 256, 512]
        inputs = Input((self.input_size, self.input_size, 3))

        c0 = Conv2D(n_filters[0], (1, 1), padding="same")(inputs)
        c0 = BatchNormalization()(c0)
        c0 = positional_attention_module(c0, n_filters[0])

        c1 = stem_block(c0, n_filters[0], strides=1)

        ## Encoder
        c2 = resnet_block(c1, n_filters[1], strides=2)
        c3 = resnet_block(c2, n_filters[2], strides=2)
        c4 = resnet_block(c3, n_filters[3], strides=2)

        ## Bridge
        b1 = aspp_block(c4, n_filters[4])

        ## Decoder
        d1 = attetion_block(c3, b1)
        d1 = UpSampling2D((2, 2))(d1)
        d1 = Concatenate()([d1, c3])
        d1 = resnet_block(d1, n_filters[3])

        d2 = attetion_block(c2, d1)
        d2 = UpSampling2D((2, 2))(d2)
        d2 = Concatenate()([d2, c2])
        d2 = resnet_block(d2, n_filters[2])

        d3 = attetion_block(c1, d2)
        d3 = UpSampling2D((2, 2))(d3)
        d3 = Concatenate()([d3, c1])
        d3 = resnet_block(d3, n_filters[1])

        ## output
        outputs = aspp_block(d3, n_filters[0])
        outputs = Conv2D(1, (1, 1), padding="same")(outputs)
        outputs = Activation("sigmoid")(outputs)

        ## Model
        model = Model(inputs, outputs)
        return model

class DA_ResUnetPlusPlus:
    def __init__(self, input_size=64):
        self.input_size = input_size

    def build_model(self):
        n_filters = [32, 64, 128, 256, 512]
        inputs = Input((self.input_size, self.input_size, 3))

        c0 = Conv2D(n_filters[0], (1, 1), padding="same")(inputs)
        c0 = BatchNormalization()(c0)
        c0 = dual_attetion_block(c0, n_filters[0])
        
        c1 = stem_block(c0, n_filters[0], strides=1)

        ## Encoder
        c2 = resnet_block(c1, n_filters[1], strides=2)
        c3 = resnet_block(c2, n_filters[2], strides=2)
        c4 = resnet_block(c3, n_filters[3], strides=2)

        ## Bridge
        b1 = aspp_block(c4, n_filters[4])

        ## Decoder
        d1 = attetion_block(c3, b1)
        d1 = UpSampling2D((2, 2))(d1)
        d1 = Concatenate()([d1, c3])
        d1 = resnet_block(d1, n_filters[3])

        d2 = attetion_block(c2, d1)
        d2 = UpSampling2D((2, 2))(d2)
        d2 = Concatenate()([d2, c2])
        d2 = resnet_block(d2, n_filters[2])

        d3 = attetion_block(c1, d2)
        d3 = UpSampling2D((2, 2))(d3)
        d3 = Concatenate()([d3, c1])
        d3 = resnet_block(d3, n_filters[1])

        ## output
        outputs = aspp_block(d3, n_filters[0])
        outputs = Conv2D(1, (1, 1), padding="same")(outputs)
        outputs = Activation("sigmoid")(outputs)

        ## Model
        model = Model(inputs, outputs)
        return model
    
class DoubleDA_ResUnetPlusPlus:
    def __init__(self, input_size=64):
        self.input_size = input_size

    def build_model(self):
        n_filters = [32, 64, 128, 256, 512]
        inputs = Input((self.input_size, self.input_size, 3))

        c0 = Conv2D(n_filters[0], (1, 1), padding="same")(inputs)
        c0 = BatchNormalization()(c0)
        c0 = dual_attetion_block(c0, n_filters[0])

        c1 = stem_block(c0, n_filters[0], strides=1)

        ## Encoder
        c2 = resnet_block(c1, n_filters[1], strides=2)
        c3 = resnet_block(c2, n_filters[2], strides=2)
        c4 = resnet_block(c3, n_filters[3], strides=2)

        ## Bridge
        b1 = aspp_block(c4, n_filters[4])

        ## Decoder
        d1 = attetion_block(c3, b1)
        d1 = UpSampling2D((2, 2))(d1)
        d1 = Concatenate()([d1, c3])
        d1 = resnet_block(d1, n_filters[3])

        d2 = attetion_block(c2, d1)
        d2 = UpSampling2D((2, 2))(d2)
        d2 = Concatenate()([d2, c2])
        d2 = resnet_block(d2, n_filters[2])

        d3 = attetion_block(c1, d2)
        d3 = UpSampling2D((2, 2))(d3)
        d3 = Concatenate()([d3, c1])
        d3 = resnet_block(d3, n_filters[1])

        ## output
        outputs = aspp_block(d3, n_filters[0])
        outputs = dual_attetion_block(outputs, outputs.shape.as_list()[-1])
        outputs = Conv2D(1, (1, 1), padding="same")(outputs)
        outputs = Activation("sigmoid")(outputs)

        ## Model
        model = Model(inputs, outputs)
        return model