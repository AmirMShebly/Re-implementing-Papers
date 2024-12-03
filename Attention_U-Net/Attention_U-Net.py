import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, concatenate, Activation
)
from tensorflow.keras.models import Model


def attention_gate(skip_input, gating_input, n_filters):
    theta_g = Conv2D(n_filters, kernel_size=1, strides=1, padding="same")(gating_input)
    phi_x = Conv2D(n_filters, kernel_size=1, strides=1, padding="same")(skip_input)

    attention_add = tf.keras.layers.add([theta_g, phi_x])
    attention_relu = Activation('relu')(attention_add)
    psi = Conv2D(1, kernel_size=1, strides=1, padding="same")(attention_relu)
    attention_coefficients = Activation('sigmoid')(psi)

    attention_output = tf.keras.layers.multiply([skip_input, attention_coefficients])
    return attention_output


def attention_unet_model(input_size=(96, 128, 3), n_filters=32, n_classes=23):
    inputs = Input(input_size)

    # Encoder
    c1 = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    c1 = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(c1)
    p1 = MaxPooling2D(pool_size=(2, 2))(c1)

    c2 = Conv2D(n_filters * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(p1)
    c2 = Conv2D(n_filters * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(c2)
    p2 = MaxPooling2D(pool_size=(2, 2))(c2)

    c3 = Conv2D(n_filters * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(p2)
    c3 = Conv2D(n_filters * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(c3)
    p3 = MaxPooling2D(pool_size=(2, 2))(c3)

    c4 = Conv2D(n_filters * 8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(p3)
    c4 = Conv2D(n_filters * 8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    p4 = Dropout(0.3)(p4)

    c5 = Conv2D(n_filters * 16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(p4)
    c5 = Conv2D(n_filters * 16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(c5)
    c5 = Dropout(0.3)(c5)

    u6 = Conv2DTranspose(n_filters * 8, 3, strides=(2, 2), padding="same")(c5)
    att6 = attention_gate(c4, u6, n_filters * 8)
    u6 = concatenate([u6, att6])
    c6 = Conv2D(n_filters * 8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(u6)
    c6 = Conv2D(n_filters * 8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(c6)

    u7 = Conv2DTranspose(n_filters * 4, 3, strides=(2, 2), padding="same")(c6)
    att7 = attention_gate(c3, u7, n_filters * 4)
    u7 = concatenate([u7, att7])
    c7 = Conv2D(n_filters * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(u7)
    c7 = Conv2D(n_filters * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(c7)

    u8 = Conv2DTranspose(n_filters * 2, 3, strides=(2, 2), padding="same")(c7)
    att8 = attention_gate(c2, u8, n_filters * 2)
    u8 = concatenate([u8, att8])
    c8 = Conv2D(n_filters * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(u8)
    c8 = Conv2D(n_filters * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(c8)

    u9 = Conv2DTranspose(n_filters, 3, strides=(2, 2), padding="same")(c8)
    att9 = attention_gate(c1, u9, n_filters)
    u9 = concatenate([u9, att9])
    c9 = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(u9)
    c9 = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(c9)

    outputs = Conv2D(n_classes, 1, activation='softmax', padding='same')(c9)

    model = Model(inputs=inputs, outputs=outputs)

    return model
