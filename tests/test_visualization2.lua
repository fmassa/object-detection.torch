require 'cutorch'
require 'nnf'
require 'cudnn'
require 'inn'
dofile 'visualize_detections.lua'

cutorch.setDevice(2)

--model = torch.load('cachedir/test2_frcnn/model.t7')
model = torch.load('cachedir/model.t7')
--model:add(nn.SoftMax():cuda())

image_transformer= nnf.ImageTransformer{mean_pix={102.9801,115.9465,122.7717},
                                              raw_scale = 255,
                                              swap = {3,2,1}}


ds = nnf.DataSetPascal{image_set='test',
                         datadir='datasets/VOCdevkit',
                         roidbdir='data/selective_search_data'
                         }

fp = nnf.FRCNN{image_transformer=image_transformer}
fp:evaluate()
model:evaluate()
detect = nnf.ImageDetect(model,fp)

im_idx = 700

I = ds:getImage(im_idx)
boxes = ds:getROIBoxes(im_idx)
--boxes = ds:getGTBoxes(im_idx)

scores,bb = detect:detect(I,boxes)

w = visualize_detections(I,boxes,scores,0.5,ds.classes)

Im = w:image()
II = Im:toFloatTensor()

image.save('example_frcnn.jpg',II)

