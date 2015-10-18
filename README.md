## Object detection in torch

This library aims to provide a simple architecture to easily perform object detection in [torch](http://torch.ch).
It currently contains code for training the following frameworks: [RCNN](http://arxiv.org/abs/1311.2524), [SPP](http://arxiv.org/abs/1406.4729) and [Fast-RCNN](http://arxiv.org/abs/1504.08083).

It consists of 6 basic classes:

* ImageTransformer: Preprocess an image before feeding it to the network
* DataSetDetection: Generic dataset class for object detection.
  * DataSetPascal
  * DataSetCOCO (not finished)
* [FeatureProvider](#feat_provider): Implements the necessary operations on images and bounding boxes
  * [RCNN](#rcnn)
  * [SPP](#spp)
  * [Fast-RCNN](#frcnn)
* [BatchProvider](#batch_provider): Samples random patches
  * [BatchProviderRC](#batch_provider_rc): ROI-Centric
  * [BatchProviderIC](#batch_provider_ic): Image-Centric
* ImageDetect: Encapsulates a model and a feature provider to perform the detection
* Tester: Evaluate the detection using Pascal VOC approach.

<a name="feat_provider"></a>
### Feature Provider
The `FeatureProvider` class defines the way different algorithms process an image and a set of bounding boxes to feed it to the CNN.
It implements a `getFeature(image,boxes)` function, which computes the necessary transformations in the input data, a `postProcess()`, which takes the output of the network plus the original inputs and post-process them. This post-processing could be a bounding box regression step, for example.

<a name="rcnn"></a>
#### RCNN
This is the first work that used CNNs for object detection using bounding box proposals.
The transformation is the simplest one. It crops the image at the specified positions given by the bounding boxes, and rescale them to be square.
<a name="spp"></a>
#### SPP
Contrary to RCNN, SPP crops the images in the feature space (here, `conv5`). It allows to compute the convolutional features once for the entire image, making it much more efficient.
<a name="frcnn"></a>
#### Fast-RCNN
Similar to SPP, Fast-RCNN also crops the images in the feature space, but instead of keeping the convolutional layers fixed, they allow it to train together with the fully-connected layers.


<a name="batch_provider"></a>
### Batch Provider
This class implements sampling strategies for training Object Detectors.
In its constructor, it takes as argument a `DataSetDetection`, and a `FeatureProvider`.
It implements a `getBatch` function, which samples from the `DataSet` using `FeatureProvider`.

<a name="batch_provider_rc"></a>
#### BatchProviderRC
ROI-Centric Batch Provider, it samples the patches randomly over all the pool of patches.

<a name="batch_provider_ic"></a>
#### BatchProviderIC
Image-Centric Batch Provider, it first samples a set of images, and then a set of patches is sampled on those sampled images.

### Examples
Here we show a simple example demonstrating how to perform object detection given an image and a set of bounding boxes. 

```lua
require 'nnf'
require 'image'
require 'nn'

model = torch.load('model.t7')
I = image.lena()
bboxes = {1,1,200,200}

image_transformer= nnf.ImageTransformer{mean_pix={102.9801,115.9465,122.7717},
                                        raw_scale = 255,
                                        swap = {3,2,1}}
feat_provider = nnf.RCNN{crop_size=227,image_transformer=image_transformer}

-- the following could also be done by creating an instance of ImageDetect
-- and calling :detect(I,boxes)
feats = feat_provider:getFeature(I,bboxes)
scores = feat_provider:compute(model,feats)

-- visualization
threshold = 0.5
visualize_detections(I,bboxes,scores,threshold)

```

For an illustration on how to use this code to train a detector, or to evaluate it on Pascal, see the [examples](http://github.com/fmassa/object-detection.torch/tree/master/examples).

#### Bounding box proposals
Note that this repo doesn't contain code for generating bounding box proposals. For the moment, they are pre-computed and loaded at run time.

### Dependencies

It requires the following packages

 - [xml](http://doc.lubyk.org/xml.html) (For `DataSetPascal`)
 - [matio-ffi.torch](https://github.com/soumith/matio-ffi.torch) (For `DataSetPascal`)
 - [hdf5](https://github.com/deepmind/torch-hdf5) (for `SPP`)
 - [inn](https://github.com/szagoruyko/imagine-nn) (for `SPP`)

To install them all, do

```
## xml
luarocks install xml

## matio
# OSX
brew install libmatio
# Ubuntu
sudo apt-get install libmatio2

luarocks install matio
```

To install `hdf5`, follow the instructions in [here](https://github.com/deepmind/torch-hdf5/blob/master/doc/usage.md)

### Old code
The old version of this repo can be found [here](https://github.com/fmassa/object-detection.torch/tree/legacy).


### Running this code

First, clone this repo
```
git clone https://github.com/fmassa/object-detection.torch.git
```

The zeiler pretrained model is available at [https://drive.google.com/open?id=0B-TTdm1WNtybdzdMUHhLc05PSE0&authuser=0](https://drive.google.com/open?id=0B-TTdm1WNtybdzdMUHhLc05PSE0&authuser=0).
It is supposed to be at `data/models`.
If you want to use your own model in SPP framework, make sure that it follows the pattern
```
model = nn.Sequential()
model:add(features)
model:add(pooling_layer)
model:add(classifier)
```
where `features` can be a `nn.Sequential` of several convolutions and `pooling_layer` is the last pooling with reshaping of the data to feed it to the classifer. See `models/zeiler.lua` for an example.


The default is to consider that the dataset is present in `datasets/VOCdevkit/VOC2007/`.
The default location of bounding boxes `.mat` files (in RCNN format) is supposed to be in `data/selective_search_data/`.

