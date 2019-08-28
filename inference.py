from tensorflow.keras.models import load_model
from utils import get_image, preprocess, deprocess
import numpy as np
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.models import model_from_json
import tensorflow as tf
from adain import AdaIN

ALPHA = 0.75
MODEL_PATH = "saved/test.h5"
CONTENT_PATH = "images\content\stata.jpg"
STYLE_PATH = "images\style\cat.jpg"

def main():  
    # Load model
    model = load_model(MODEL_PATH, custom_objects={'AdaIN': AdaIN})

    # Get content image
    content = get_image(CONTENT_PATH, resize=False)
    content = preprocess(content)
    content = np.expand_dims(content, axis=0)

    # Get style image
    style = get_image(STYLE_PATH, resize=False)
    style = preprocess(style)
    style = np.expand_dims(style, axis=0)

    # Set alpha Value
    alpha = tf.convert_to_tensor(ALPHA)  # 0 < alpha <= 1
    alpha = np.expand_dims(alpha, axis=0)

    # Do inference
    y = model.predict([content, style, alpha])[0]

    # Convert output array to image
    y = np.squeeze(y, axis=0)
    y = deprocess(y)
    img = array_to_img(y)

    # Show image
    img.show(command='fim')


if __name__ == '__main__':
    main()
    
