require 'nnf'

dt = torch.load('pascal_2007_train.t7')
ds = nnf.DataSetPascal{image_set='train',
                       datadir='/home/francisco/work/datasets/VOCdevkit',
                       roidbdir='/home/francisco/work/datasets/rcnn/selective_search_data'
                       }
if false then
  ds.roidb = {}
  for i=1,ds:size() do
    ds.roidb[i] = torch.IntTensor(10,4):random(1,5)
    ds.roidb[i][{{},{3,4}}]:add(6)
  end
else
  ds.roidb = dt.roidb
end

if false then
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
  
  features:add(nn.SpatialConvolutionMM(3,96,11,11,4,4,5,5))
  features:add(nn.ReLU(true))
  features:add(nn.SpatialConvolutionMM(96,128,5,5,2,2,2,2))
  features:add(nn.ReLU(true))
  features:add(nn.SpatialMaxPooling(2,2,2,2))

  classifier:add(nn.Linear(128*7*7,1024))
  classifier:add(nn.ReLU(true))
  classifier:add(nn.Dropout(0.5))
  classifier:add(nn.Linear(1024,21))

  local prl = nn.ParallelTable()
  prl:add(features)
  prl:add(nn.Identity())
  model:add(prl)
  model:add(nnf.ROIPooling(7,7):setSpatialScale(1/16))
  model:add(nn.View(-1):setNumInputDims(3))
  model:add(classifier)

end

parameters,gradParameters = model:getParameters()

optimState = {learningRate = 1e-2, weightDecay = 0.0005, momentum = 0.9,
              learningRateDecay = 0}

--------------------------------------------------------------------------
-- training
--------------------------------------------------------------------------

model:float()
model:training()

criterion = nn.CrossEntropyCriterion():float()

max_iter = 20

function train()
  local err = 0
  for i=1,max_iter do
    xlua.progress(i,max_iter)
    inputs,target = bp:getBatch(inputs,target)
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
  print('Training error: '..err/max_iter)
end

train()

if false then
  m = nnf.ROIPooling(50,50):float()
  o = m:forward(batches)
  g = m:backward(batches,o)
end

