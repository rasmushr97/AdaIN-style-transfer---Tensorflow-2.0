import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, UpSampling2D, Reshape, MaxPooling2D, Lambda, Layer
from tensorflow.keras.models import Sequential, Model
from adain import AdaIN

def build_vgg19(input_shape, weights_path):
    weights = np.load(weights_path)

    # Create vgg19 structure
    model = Sequential([
        Conv2D(64, (3,3), activation='relu', input_shape=input_shape, padding='same', name='conv1_1'),
        Conv2D(64, (3,3), activation='relu', padding='same', name='conv1_2'),
        MaxPooling2D((2,2), name="pool1"),
        Conv2D(128, (3,3), activation='relu', padding='same', name='conv2_1'),
        Conv2D(128, (3,3), activation='relu', padding='same', name='conv2_2'),
        MaxPooling2D((2,2), name="pool2"),
        Conv2D(256, (3,3), activation='relu', padding='same', name='conv3_1'),
        Conv2D(256, (3,3), activation='relu', padding='same', name='conv3_2'),
        Conv2D(256, (3,3), activation='relu', padding='same', name='conv3_3'),
        Conv2D(256, (3,3), activation='relu', padding='same', name='conv3_4'),
        MaxPooling2D((2,2), name="pool3"),
        Conv2D(512, (3,3), activation='relu', padding='same', name='conv4_1')
    ])

    # Load in weights
    i = 0
    for layer in model.layers:
        kind = layer.name[:4]
        if kind == 'conv':
            kernel = weights['arr_%d' % i].transpose([2, 3, 1, 0])
            kernel = kernel.astype(np.float32)

            bias = weights['arr_%d' % (i + 1)]
            bias = bias.astype(np.float32)
            
            layer.set_weights([kernel, bias])

            i += 2
    
    model.trainable = False
    return model


def build_vgg19_relus(vgg19):
    relus = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1']
    features = [vgg19.get_layer(relu).output for relu in relus]
    vgg19_relus = Model(inputs=vgg19.input, outputs=features)
    vgg19_relus.trainable = False
    return vgg19_relus


def build_decoder(input_shape):
    return Sequential([
            Conv2D(256, (3,3), input_shape=input_shape, activation='relu', padding='same'),
            UpSampling2D((2,2)),
            Conv2D(256, (3,3), activation='relu', padding='same'),
            Conv2D(256, (3,3), activation='relu', padding='same'),
            Conv2D(256, (3,3), activation='relu', padding='same'),
            Conv2D(128, (3,3), activation='relu', padding='same'),
            UpSampling2D((2,2)),
            Conv2D(128, (3,3), activation='relu', padding='same'),
            Conv2D(64, (3,3), activation='relu', padding='same'),
            UpSampling2D((2,2)),
            Conv2D(64, (3,3), activation='relu', padding='same'),
            Conv2D(3, (3,3), padding='same'),
        ])


def build_model(encoder, decoder, input_shape):
    content = Input(shape=input_shape, name='content')
    style = Input(shape=input_shape, name = 'style')
    alpha = Input(shape=(1,), name='alpha')

    enc_content = encoder(content)
    enc_style = encoder(style)

    adain = AdaIN()([enc_content, enc_style, alpha])

    out = decoder(adain)

    return Model(inputs=[content, style, alpha], outputs=[out, adain])
