require 'nnf'
require 'inn'
require 'cudnn'
require 'loadcaffe'

cutorch.setDevice(2)

ds = nnf.DataSetPascal{image_set='trainval',
                       datadir='datasets/VOCdevkit',
                       roidbdir='data/selective_search_data'
                       }
local image_transformer= nnf.ImageTransformer{mean_pix={102.9801,115.9465,122.7717},
                                              raw_scale = 255,
                                              swap = {3,2,1}}

fp = nnf.RCNN{image_transformer=image_transformer,
              crop_size=224}
fp:training()
--------------------------------------------------------------------------------
-- define batch providers
--------------------------------------------------------------------------------

bp = nnf.BatchProvider{dataset=ds,feat_provider=fp,
                       bg_threshold={0.0,0.5},
                       nTimesMoreData=2,
                       iter_per_batch=100,
                      }
bp:setupData()

--------------------------------------------------------------------------------
-- define model
--------------------------------------------------------------------------------
model = nn.Sequential()
do 
  --[[
  local rcnnfold = '/home/francisco/work/projects/object-detection.torch/data/models/imagenet_models/'
  local base_model = loadcaffe.load(
  rcnnfold..'CaffeNet_train.prototxt',
  rcnnfold..'CaffeNet.v2.caffemodel',
  'cudnn')
  for i=1,14 do
    features:add(base_model:get(i):clone())
  end
  for i=17,22 do
    classifier:add(base_model:get(i):clone())
  end
  local linear = nn.Linear(4096,21):cuda()
  linear.weight:normal(0,0.01)
  linear.bias:zero()
  classifier:add(linear)
  --]]
  local features = nn.Sequential()
  local classifier = nn.Sequential()
  local fold = 'data/models/imagenet_models/alexnet/'
  local m1 = torch.load(fold..'features.t7')
  local m2 = torch.load(fold..'top.t7')
    for i=1,14 do
      features:add(m1:get(i):clone())
    end
    features:get(3).padW = 1
    features:get(3).padH = 1
    features:get(7).padW = 1
    features:get(7).padH = 1
    for i=2,7 do
      classifier:add(m2:get(i):clone())
    end
    local linear = nn.Linear(4096,21):cuda()
    linear.weight:normal(0,0.01)
    linear.bias:zero()
    classifier:add(linear)
  collectgarbage()
  --local prl = nn.ParallelTable()
  --prl:add(features)
  --prl:add(nn.Identity())
  --model:add(prl)
  --model:add(nnf.ROIPooling(6,6):setSpatialScale(1/16))
  --model:add(inn.ROIPooling(6,6):setSpatialScale(1/16))
  model:add(features)
  model:add(nn.SpatialAdaptiveMaxPooling(6,6))
  model:add(nn.View(-1):setNumInputDims(3))
  model:add(classifier)
end
model:cuda()
--------------------------------------------------------------------------------
-- train
--------------------------------------------------------------------------------

criterion = nn.CrossEntropyCriterion():cuda()

trainer = nnf.Trainer(model,criterion,bp)

for i=1,400 do
  if i == 300 then
    trainer.optimState.learningRate = trainer.optimState.learningRate/10
  end
  print(('Iteration %3d/%-3d'):format(i,400))
  trainer:train(100)
end

--------------------------------------------------------------------------------
-- evaluate
--------------------------------------------------------------------------------

-- add softmax to classfier
model:add(nn.SoftMax():cuda())

dsv = nnf.DataSetPascal{image_set='test',
                         datadir='datasets/VOCdevkit',
                         roidbdir='data/selective_search_data'
                         }


fpv = nnf.RCNN{image_transformer=image_transformer,
               crop_size=224}
fpv:evaluate()
exp_name = 'test1_rcnn'

tester = nnf.Tester(model,fpv,dsv)
tester.cachefolder = 'cachedir/'..exp_name
tester:test(40000)
