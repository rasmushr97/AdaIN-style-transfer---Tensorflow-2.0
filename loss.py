import tensorflow as tf
import numpy as numpy
from utils import deprocess, preprocess

def get_loss(encoder, vgg19_relus, epsilon=1e-5, style_weight=1.0, color_weight=0.0):

    def loss(y_true, y_pred):
        # y_true == input == [content, style]
        out, adain = y_pred[0], y_pred[1]

        # Encode output and compute content_loss
        out = deprocess(out)
        out = preprocess(out)
        enc_out = encoder(out)
        content_loss = tf.reduce_sum(tf.reduce_mean(tf.square(enc_out - adain), axis=[1, 2]))
        
        # Compute style loss from vgg relus
        style = y_true[1]
        style_featues = vgg19_relus(style)
        gen_features = vgg19_relus(out)
        style_layer_loss = []
        for enc_style_feat, enc_gen_feat in zip(style_featues, gen_features):
            meanS, varS = tf.nn.moments(enc_style_feat, [1, 2])
            meanG, varG = tf.nn.moments(enc_gen_feat,   [1, 2])

            sigmaS = tf.sqrt(varS + epsilon)
            sigmaG = tf.sqrt(varG + epsilon)

            l2_mean  = tf.reduce_sum(tf.square(meanG - meanS))
            l2_sigma = tf.reduce_sum(tf.square(sigmaG - sigmaS))

            style_layer_loss.append(l2_mean + l2_sigma)

        style_loss = tf.reduce_sum(style_layer_loss)

        # Compute color loss
        style_color_mean, style_color_var = tf.nn.moments(style, [1, 2])
        gen_color_mean, gen_color_var = tf.nn.moments(out, [1, 2])

        color_sigmaS = tf.sqrt(style_color_var)
        color_sigmaG = tf.sqrt(gen_color_var)

        l2_mean  = tf.reduce_sum(tf.square(gen_color_mean - style_color_mean))
        l2_sigma = tf.reduce_sum(tf.square(color_sigmaG - color_sigmaS))

        color_loss = l2_mean + l2_sigma

        # Compute the total loss
        weighted_style_loss = style_weight * style_loss
        weighted_color_loss = color_loss * color_weight
        total_loss = content_loss + weighted_style_loss + weighted_color_loss
        return total_loss, content_loss, weighted_style_loss, weighted_color_loss
    
    return loss