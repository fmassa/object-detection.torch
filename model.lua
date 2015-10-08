require 'nn'
require 'inn'
require 'cudnn'
local reshapeLastLinearLayer = paths.dofile('utils.lua').reshapeLastLinearLayer
local convertCaffeModelToTorch = paths.dofile('utils.lua').convertCaffeModelToTorch

-- 1.1. Create Network
local config = opt.netType
local createModel = paths.dofile('models/' .. config .. '.lua')
print('=> Creating model from file: models/' .. config .. '.lua')
model = createModel(opt.backend)

-- convert to accept inputs in the range 0-1 RGB format
convertCaffeModelToTorch(model,{1,1})

reshapeLastLinearLayer(model,#classes+1)
image_mean = {128/255,128/255,128/255}

if opt.algo == 'RCNN' then
  classifier = model
elseif opt.algo == 'SPP' then
  features = model:get(1)
  classifier = model:get(3)
elseif opt.algo == 'FRCNN' then
  local temp = nn.Sequential()
  local features = model:get(1)
  local classifier = model:get(3)
  local prl = nn.ParallelTable()
  prl:add(features)
  prl:add(nn.Identity())
  temp:add(prl)
  temp:add(nnf.ROIPooling(7,7))
  temp:add(nn.View(-1):setNumInputDims(3))
  temp:add(classifier)
end

-- 2. Create Criterion
criterion = nn.CrossEntropyCriterion()

print('=> Model')
print(model)

print('=> Criterion')
print(criterion)

-- 3. If preloading option is set, preload weights from existing models appropriately
if opt.retrain ~= 'none' then
  assert(paths.filep(opt.retrain), 'File not found: ' .. opt.retrain)
  print('Loading model from file: ' .. opt.retrain);
  classifier = torch.load(opt.retrain)
end

-- 4. Convert model to CUDA
print('==> Converting model to CUDA')
model = model:cuda()
criterion:cuda()

collectgarbage()



