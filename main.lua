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

trainer = nnf.Trainer(classifier,criterion)
trainer.optimState.learningRate = opt.lr

local conf_classes = {}
table.insert(conf_classes,'background')
for i=1,#classes do
  table.insert(conf_classes,classes[i])
end
trainer.confusion = optim.ConfusionMatrix(conf_classes)

validator = nnf.Tester(classifier,feat_provider_test)
validator.cachefolder = opt.save_base
validator.cachename = 'validation_data.t7'
validator.batch_provider = batch_provider_test

logger = optim.Logger(paths.concat(opt.save,'log.txt'))
val_err = {}
val_counter = 0
reduc_counter = 0

inputs = torch.FloatTensor()
targets = torch.IntTensor()
for i=1,opt.num_iter do

  print('Iteration: '..i..'/'..opt.num_iter)
  inputs,targets = batch_provider:getBatch(inputs,targets)
  print('==> Training '..paths.basename(opt.save_base))
  trainer:train(inputs,targets)
  print('==> Training Error: '..trainer.fx[i])
  print(trainer.confusion)

  collectgarbage() 

  err = validator:validate(criterion)
  print('==> Validation Error: '..err)
  table.insert(val_err,err)

  logger:add{['train error (iters per batch='..batch_provider.iter_per_batch..
              ')']=trainer.fx[i],['val error']=err,
              ['learning rate']=trainer.optimState.learningRate}

  val_counter = val_counter + 1

  local val_err_t = torch.Tensor(val_err)
  local _,lmin = val_err_t:min(1)
  if val_counter-lmin[1] >= opt.nsmooth then
    print('Reducing learning rate')
    trainer.optimState.learningRate = trainer.optimState.learningRate/2
    if opt.nildfdx == true then
      trainer.optimState.dfdx= nil
    end
    val_counter = 0
    val_err = {}
    reduc_counter = reduc_counter + 1
    if reduc_counter >= opt.nred then
      print('Stopping training at iteration '..i)
      break
    end
  end

  collectgarbage()
  collectgarbage()
  --sanitize(model)
  --torch.save(paths.concat(opt.save, 'model_' .. epoch .. '.t7'), classifier)
  --torch.save(paths.concat(opt.save, 'optimState_' .. epoch .. '.t7'), trainer.optimState)
end

--sanitize(classifier)
torch.save(paths.concat(opt.save, 'model.t7'), classifier)

ds_train.roidb = nil
collectgarbage()
collectgarbage()

--------------------------------------------------------------------------------
-- Do full evaluation
--------------------------------------------------------------------------------

print('==> Evaluation')
tester = nnf.Tester(classifier,feat_provider_test)
tester.cachefolder = paths.concat(opt.save,'evaluation',ds_test.dataset_name)


tester:test(opt.num_iter)

