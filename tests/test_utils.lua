require 'nnf'
require 'nn'

function getDS()
  local dt = torch.load('pascal_2007_train.t7')
  local ds = nnf.DataSetPascal{image_set='train',
                             datadir='/home/francisco/work/datasets/VOCdevkit',
                             roidbdir='/home/francisco/work/datasets/rcnn/selective_search_data'
                             }
  ds.roidb = dt.roidb
  return ds
end

function getModel()
  local features = nn.Sequential()
  features:add(nn.SpatialConvolutionMM(3,16,11,11,16,16,5,5))
  local classifier = nn.Sequential()
  classifier:add(nn.Linear(7*7*16,21))
  local model1 = nn.Sequential()
  model1:add(features)
  model1:add(nn.SpatialMaxPooling(2,2,2,2))
  model1:add(nn.View(-1):setNumInputDims(3))
  model1:add(classifier)
  local model = nn.Sequential()
  local prl = nn.ParallelTable()
  prl:add(features)
  prl:add(nn.Identity())
  model:add(prl)
  model:add(nnf.ROIPooling(7,7):setSpatialScale(1/16))
  model:add(nn.View(-1):setNumInputDims(3))
  model:add(classifier)
  return model1, model, features, classifier
end

--------------------------------------------------------------------------------
-- define dataset, models and feature providers
--------------------------------------------------------------------------------

ds = getDS()

model1, model, features, classifier = getModel()
  
fp1 = nnf.RCNN()
fp2 = nnf.FRCNN()
fp3 = nnf.SPP(features)
fp3.use_cache = false
fp3:evaluate()


