## Object detection in torch

Implementation of some object detection frameworks in [torch](torch.ch).

### Dependencies

It requires the following packages

 - [xml](http://doc.lubyk.org/xml.html)
 - [matio-ffi.torch](https://github.com/soumith/matio-ffi.torch)
 - [hdf5](https://github.com/deepmind/torch-hdf5)

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


