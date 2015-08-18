require 'nnf'

dt = torch.load('pascal_2007_train.t7')
if false then
  ds = nnf.DataSetPascal{image_set='train',
                         datadir='/home/francisco/work/datasets/VOCdevkit',
                         roidbdir='/home/francisco/work/datasets/rcnn/selective_search_data'
                        }
else
  ds = nnf.DataSetPascal{image_set='train',
                         datadir='datasets/VOCdevkit',
                         roidbdir='data/selective_search_data'
                         }
end

if false then
  ds.roidb = {}
  for i=1,ds:size() do
    ds.roidb[i] = torch.IntTensor(10,4):random(1,5)
    ds.roidb[i][{{},{3,4}}]:add(6)
  end
else
  ds.roidb = dt.roidb
end

if true then
  bp = nnf.BatchProviderROI(ds)
  bp:setupData()
else
  bp = nnf.BatchProviderROI(ds)
  local temp = torch.load('pascal_2007_train_bp.t7')
  bp.bboxes = temp.bboxes
end

---------------------------------------------------------------------------------------
-- model
---------------------------------------------------------------------------------------
do

  model = nn.Sequential()
  local features = nn.Sequential()
  local classifier = nn.Sequential()

  if false then
    features:add(nn.SpatialConvolutionMM(3,96,11,11,4,4,5,5))
    features:add(nn.ReLU(true))
    features:add(nn.SpatialConvolutionMM(96,128,5,5,2,2,2,2))
    features:add(nn.ReLU(true))
    features:add(nn.SpatialMaxPooling(2,2,2,2))

    classifier:add(nn.Linear(128*7*7,1024))
    classifier:add(nn.ReLU(true))
    classifier:add(nn.Dropout(0.5))
    classifier:add(nn.Linear(1024,21))
  
  else
    require 'loadcaffe'
--    local rcnnfold = '/home/francisco/work/libraries/rcnn/'
--    local base_model = loadcaffe.load(
--          rcnnfold..'model-defs/pascal_finetune_deploy.prototxt',
--          rcnnfold..'data/caffe_nets/finetune_voc_2012_train_iter_70k',
--    'cudnn')

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
  end
  collectgarbage()

  local prl = nn.ParallelTable()
  prl:add(features)
  prl:add(nn.Identity())
  model:add(prl)
  model:add(nnf.ROIPooling(6,6):setSpatialScale(1/16))
  --model:add(inn.ROIPooling(6,6):setSpatialScale(1/16))
  model:add(nn.View(-1):setNumInputDims(3))
  model:add(classifier)

end
print(model)
parameters,gradParameters = model:getParameters()

optimState = {learningRate = 1e-3, weightDecay = 0.0005, momentum = 0.9,
              learningRateDecay = 0}

--------------------------------------------------------------------------
-- training
--------------------------------------------------------------------------

model:cuda()
model:training()

criterion = nn.CrossEntropyCriterion():cuda()

display_iter = 20

inputs = {torch.CudaTensor(),torch.FloatTensor()}
target = torch.CudaTensor()

function train()
  local err = 0
  for i=1,display_iter do
    xlua.progress(i,display_iter)
    inputs0,target0 = bp:getBatch(inputs0,target0)
    inputs[1]:resize(inputs0[1]:size()):copy(inputs0[1])
    inputs[2]:resize(inputs0[2]:size()):copy(inputs0[2])
    target:resize(target0:size()):copy(target0)
    local batchSize = target:size(1)

    local feval = function(x)
      if x ~= parameters then
        parameters:copy(x)
      end
      gradParameters:zero()

      local outputs = model:forward(inputs)

      local f = criterion:forward(outputs,target)
      local df_do = criterion:backward(outputs,target)

      model:backward(inputs,df_do)

      if normalize then
        gradParameters:div(batchSize)
        f = f/batchSize
      end

      return f,gradParameters
    end

    local x,fx = optim.sgd(feval,parameters,optimState)
    err = err + fx[1]
  end
  print('Training error: '..err/display_iter)
end

stepsize = 30000

num_iter = 3000

for i=1,num_iter do
  print(('Iteration: %d/%d'):format(i,num_iter))
  if i%(stepsize/display_iter) == 0 then
    optimState.learningRate = optimState.learningRate/10
  end

  train()
  
end
