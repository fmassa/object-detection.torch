require 'nnf'
require 'loadcaffe'

ds = nnf.DataSetPascal{image_set='trainval',
                       datadir='datasets/VOCdevkit',
                       roidbdir='data/selective_search_data'
                       }
local image_transformer= nnf.ImageTransformer{mean_pix={102.9801,115.9465,122.7717},
                                              raw_scale = 255,
                                              swap = {3,2,1}}

fp = nnf.FRCNN{image_transformer=image_transformer}
fp:training()
--------------------------------------------------------------------------------
-- define batch providers
--------------------------------------------------------------------------------

bp = nnf.BatchProviderROI{dataset=ds,feat_provider=fp,
                          bg_threshold={0.1,0.5}
                         }
bp:setupData()

--------------------------------------------------------------------------------
-- define model
--------------------------------------------------------------------------------
model = nn.Sequential()
do 
  local rcnnfold = '/home/francisco/work/libraries/caffe/examples/imagenet/'
  local base_model = loadcaffe.load(
  rcnnfold..'imagenet_deploy.prototxt',
  rcnnfold..'caffe_reference_imagenet_model',
  'cudnn')

  for i=1,14 do
    features:add(base_model:get(i):clone())
  end
  for i=17,22 do
    classifier:add(base_model:get(i):clone())
  end
  classifier:add(nn.Linear(4096,21):cuda())

  collectgarbage()
  local prl = nn.ParallelTable()
  prl:add(features)
  prl:add(nn.Identity())
  model:add(prl)
  --model:add(nnf.ROIPooling(6,6):setSpatialScale(1/16))
  model:add(inn.ROIPooling(6,6):setSpatialScale(1/16))
  model:add(nn.View(-1):setNumInputDims(3))
  model:add(classifier)

end

--------------------------------------------------------------------------------
-- train
--------------------------------------------------------------------------------

criterion = nn.CrossEntropyCriterion()

trainer = nnf.Trainer(model,criterion,bp)

for i=1,10 do
  trainer:train(10)
end

--------------------------------------------------------------------------------
-- evaluate
--------------------------------------------------------------------------------

-- add softmax to classfier
model:add(nn.SoftMax())

dsv = nnf.DataSetPascal{image_set='test',
                         datadir='datasets/VOCdevkit',
                         roidbdir='data/selective_search_data'
                         }


fpv = nnf.FRCNN{image_transformer=image_transformer}
fpv:evaluate()

tester = nnf.Tester_FRCNN(model,fpv,dsv)
tester.cachefolder = 'cachedir/'..exp_name
tester:test(num_iter)
