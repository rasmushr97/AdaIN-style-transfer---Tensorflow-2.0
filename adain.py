from tensorflow.keras.layers import Layer
import tensorflow as tf

class AdaIN(Layer):
    def __init__(self, epsilon=1e-5, **kwargs):
        self.eps = epsilon
        super(AdaIN, self).__init__(**kwargs)

    def call(self, inputs):
        content, style, alpha = inputs

        meanC, varC = tf.nn.moments(content, [1, 2], keepdims=True)
        meanS, varS = tf.nn.moments(style,   [1, 2], keepdims=True)

        sigmaC = tf.sqrt(tf.add(varC, self.eps))
        sigmaS = tf.sqrt(tf.add(varS, self.eps))
        
        adain = (content - meanC) * sigmaS / sigmaC + meanS
        return alpha * adain + (1-alpha) * content
        