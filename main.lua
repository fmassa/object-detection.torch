require 'nnf'
--require 'cunn'
require 'optim'
require 'trepl'

local opts = paths.dofile('opts.lua')
opt = opts.parse(arg)
print(opt)

if opt.seed ~= 0 then
  torch.manualSeed(opt.seed)
  if opt.gpu > 0 then
    cutorch.manualSeed(opt.seed)
  end
end

torch.setnumthreads(opt.numthreads)

local tensor_type
if opt.gpu > 0 then
  require 'cunn'
  cutorch.setDevice(opt.gpu)
  tensor_type = 'torch.CudaTensor'
  print('Using GPU mode on device '..opt.gpu)
else
  require 'nn'
  tensor_type = 'torch.FloatTensor'
  print('Using CPU mode')
end

--------------------------------------------------------------------------------

model, criterion = paths.dofile('model.lua')
model:type(tensor_type)
criterion:type(tensor_type)

-- prepate training and test data
paths.dofile('data.lua')

-- Do training
paths.dofile('train.lua')

-- evaluation
print('==> Evaluating')
-- add softmax to classifier, because we were using nn.CrossEntropyCriterion
local softmax = nn.SoftMax()
softmax:type(tensor_type)
model:add(softmax)

feat_provider:evaluate()

-- define the class to test the model on the full dataset
tester = nnf.Tester(model, feat_provider, ds_test)
tester.cachefolder = rundir
tester:test(opt.num_iter)
