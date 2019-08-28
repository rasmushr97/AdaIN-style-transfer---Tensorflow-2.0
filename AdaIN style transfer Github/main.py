from models import build_model
import tensorflow as tf
from loss import get_loss
from models import build_vgg19_relus, build_decoder, build_model, build_vgg19
from train import train
from utils import ImageDataset
import datetime

VGG_PATH = "vgg19/vgg19_normalised.npz"
STYLE_WEIGHT = 2.0
COLOR_LOSS = 0.0
BATCH_SIZE = 4
CONTENT_DS_PATH = "E:/coco/unlabeled2017"
STYLE_DS_PATH = "E:/wikiart1"
INPUT_SHAPE = (None, None, 3)
SAVE_PATH = "saved/test.h5"
EPOCHS = 5
EPSILON = 1e-5

def main():
    # Create dataset
    content_ds = ImageDataset(CONTENT_DS_PATH, batch_size=BATCH_SIZE)
    style_ds = ImageDataset(STYLE_DS_PATH, batch_size=BATCH_SIZE)

    # Build model
    vgg19 = build_vgg19(INPUT_SHAPE, VGG_PATH)  # encoder
    decoder = build_decoder(vgg19.output.shape[1:])  # input shape == encoder output shape
    model = build_model(vgg19, decoder, INPUT_SHAPE)

    #model.load_weights(SAVE_PATH)

    # Get loss
    vgg19_relus = build_vgg19_relus(vgg19)
    loss = get_loss(vgg19, vgg19_relus, epsilon=EPSILON, style_weight=STYLE_WEIGHT, color_weight=COLOR_LOSS)

    # Train model
    train(model, content_ds, style_ds, loss, n_epochs=EPOCHS, save_path=SAVE_PATH)


        
if __name__ == "__main__":
    main()