import numpy as np
import keras
import cv2

import metrics


def conv_block(layer, num_filters):
    x = keras.layers.Conv2D(num_filters, 3, padding="same")(layer)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)

    x = keras.layers.Conv2D(num_filters, 3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)

    return x


def encoder_block(layer, num_filters):
    x = conv_block(layer, num_filters)
    p = keras.layers.MaxPool2D((2, 2))(x)
    return x, p


def decoder_block(layer, skip_features, num_filters):
    x = keras.layers.Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(layer)
    x = keras.layers.Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x


def build_unet(input_shape):
    inputs = keras.layers.Input(input_shape)

    s1, p1 = encoder_block(inputs, 8)
    s2, p2 = encoder_block(p1, 16)
    s3, p3 = encoder_block(p2, 32)
    s4, p4 = encoder_block(p3, 64)

    b1 = conv_block(p4, 128)

    d1 = decoder_block(b1, s4, 64)
    d2 = decoder_block(d1, s3, 32)
    d3 = decoder_block(d2, s2, 16)
    d4 = decoder_block(d3, s1, 8)

    outputs = keras.layers.Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    model = keras.Model(inputs, outputs, name="U-Net")
    model.compile(optimizer=keras.optimizers.Adam(1e-4, decay=1e-6), loss=metrics.dice_p_bce,
                  metrics=[metrics.dice_coef, 'binary_accuracy', metrics.true_positive_rate])
    return model


def train_model(train_generator, valid_generator, epochs=5):
    segm_model = build_unet((768,768,3))
    history = segm_model.fit_generator(generator=train_generator,
                                        validation_data=valid_generator,
                                        epochs=epochs)
    return segm_model, history


def predict(model_path, img_path):
    model = keras.models.load_model(model_path, compile=False)
    img = cv2.imread(img_path)
    prediction = model.predict(np.array([img]))[0] > 0.7
    return prediction