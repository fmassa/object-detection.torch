## Object detection in torch

Implementation of some object detection frameworks in [torch](http://torch.ch).

### Note on new code
You should probably check the [refactoring branch of this repository](https://github.com/fmassa/object-detection.torch/tree/refactoring), which simplifies the code structure, and also contains an implementation of Fast-RCNN and threaded RCNN (making it much faster than standard RCNN). It will be merged to master soon.

### Dependencies

It requires the following packages

 - [xml](http://doc.lubyk.org/xml.html)
 - [matio-ffi.torch](https://github.com/soumith/matio-ffi.torch)
 - [hdf5](https://github.com/deepmind/torch-hdf5)
 - [inn](https://github.com/szagoruyko/imagine-nn)

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

To finetune the network for detection, simply run
```
th main.lua
```

To get an overview of the different parameters, do
```
th main.lua -h
```

The default is to consider that the dataset is present in `datasets/VOCdevkit/VOC2007/`.
The default location of bounding boxes `.mat` files (in RCNN format) is supposed to be in `data/selective_search_data/`.

