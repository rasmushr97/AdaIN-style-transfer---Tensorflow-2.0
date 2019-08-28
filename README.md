# AdaIN-style-transfer---Tensorflow-2.0

Created in Tensorflow 2.0
<br>
The code is based on: Huang et al. [Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization](https://arxiv.org/pdf/1703.06868.pdf) 
<br>
It borrows elements of githubs posts: <br>
- https://github.com/elleryqueenhomels/arbitrary_style_transfer <br>
- https://github.com/eridgd/AdaIN-TF <br>

# Architecture
The style transfer network follows an encoder-decoder architecure with Adaptive Instance Normalization in between the encoder and decoder.
![stn_overview](https://user-images.githubusercontent.com/13844740/33978899-d428bf2e-e0dc-11e7-9114-41b6fb8921a7.jpg)
A normalized pretrained VGG19 model is used as the encoder for this network which weigths can be found here: <br>
[Pre-trained VGG19 normalised network npz format](https://s3-us-west-2.amazonaws.com/wengaoye/vgg19_normalised.npz) (MD5 `c5c961738b134ffe206e0a552c728aea`)


The network is trained using the MS-COCO dataset the content images and the WikiArt dataset for the style images.


## Prerequisites
- Tensorflow 2.0
- Numpy
- OpenCV
- Tqdm
