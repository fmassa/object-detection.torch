require 'nnf'
require 'cunn'
require 'optim'

local opts = paths.dofile('opts.lua')
opt = opts.parse(arg)
print(opt)

if opt.seed ~= 0 then
  torch.manualSeed(opt.seed)
  cutorch.manualSeed(opt.seed)
end

cutorch.setDevice(opt.gpu)
torch.setnumthreads(opt.numthreads)

--------------------------------------------------------------------------------
-- Select target classes
--------------------------------------------------------------------------------

if opt.classes == 'all' then
  classes={'aeroplane','bicycle','bird','boat','bottle','bus','car',
           'cat','chair','cow','diningtable','dog','horse','motorbike',
           'person','pottedplant','sheep','sofa','train','tvmonitor'}
else
  classes = {opt.classes}
end

--------------------------------------------------------------------------------


paths.dofile('model.lua')
paths.dofile('data.lua')

--------------------------------------------------------------------------------
-- Prepare training model
--------------------------------------------------------------------------------
paths.dofile('train.lua')

ds_train.roidb = nil
collectgarbage()
collectgarbage()

--------------------------------------------------------------------------------
-- Do full evaluation
--------------------------------------------------------------------------------

print('==> Evaluation')
if opt.algo == 'FRCNN' then
  tester = nnf.Tester_FRCNN(model,feat_provider_test)
else
  tester = nnf.Tester(classifier,feat_provider_test)
end
tester.cachefolder = paths.concat(opt.save,'evaluation',ds_test.dataset_name)


tester:test(opt.num_iter)

