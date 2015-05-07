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

paths.dofile('model.lua')

if opt.classes == 'all' then
  classes={'aeroplane','bicycle','bird','boat','bottle','bus','car',
           'cat','chair','cow','diningtable','dog','horse','motorbike',
           'person','pottedplant','sheep','sofa','train','tvmonitor'}
else
  classes = {opt.classes}
end

--------------------------------------------------------------------------------
-- Prepare data model
--------------------------------------------------------------------------------
paths.mkdir(opt.save)

trainCache = paths.concat(opt.save_base,'trainCache.t7')
testCache = paths.concat(opt.save_base,'testCache.t7')

if paths.filep(trainCache) then
  print('Loading train metadata from cache')
  batch_provider = torch.load(trainCache)
  feat_provider = batch_provider.feat_provider
  ds_train = feat_provider.dataset
  feat_provider.model = features
else
  ds_train = nnf.DataSetPascal{image_set='trainval',classes=classes,year=opt.year,
                         datadir=opt.datadir,roidbdir=opt.roidbdir}
  
  local feat_dim
  if opt.algo == 'SPP' then
    feat_provider = nnf.SPP(ds_train)-- remove features here to reduce cache size
    feat_provider.cachedir = paths.concat(opt.cache,'features',opt.netType)
    feat_provider.scales = {600}
    feat_dim = {256*50}
  elseif opt.algo == 'RCNN' then
    feat_provider = nnf.RCNN(ds_train)
    feat_dim = {3,feat_provider.crop_size,feat_provider.crop_size}
  else
    error(("Detection framework '%s' not available"):format(opt.algo))
  end
  
  print('==> Preparing BatchProvider for training')
  batch_provider = nnf.BatchProvider(feat_provider)
  batch_provider.iter_per_batch = opt.ipb
  batch_provider.nTimesMoreData = opt.ntmd
  batch_provider.fg_fraction = opt.fg_frac
  batch_provider.bg_threshold = {0.0,0.5}
  batch_provider.do_flip = true
  batch_provider.batch_dim = feat_dim
  batch_provider:setupData()
  
  torch.save(trainCache,batch_provider)
  feat_provider.model = features
end

if paths.filep(testCache) then
  print('Loading test metadata from cache')
  batch_provider_test = torch.load(testCache)
  feat_provider_test = batch_provider_test.feat_provider
  ds_test = feat_provider_test.dataset
  feat_provider_test.model = features
else
  ds_test = nnf.DataSetPascal{image_set='test',classes=classes,year=opt.year,
                              datadir=opt.datadir,roidbdir=opt.roidbdir}
  local feat_dim
  if opt.algo == 'SPP' then
    feat_provider_test = nnf.SPP(ds_test)
    feat_provider_test.randomscale = false
    feat_provider_test.cachedir = paths.concat(opt.cache,'features',opt.netType)
    feat_provider_test.scales = {600}
    feat_dim = {256*50}
  elseif opt.algo == 'RCNN' then
    feat_provider_test = nnf.RCNN(ds_test)
    feat_dim = {3,feat_provider_test.crop_size,feat_provider_test.crop_size}
  else
    error(("Detection framework '%s' not available"):format(opt.algo))
  end
  
  print('==> Preparing BatchProvider for validation')
  batch_provider_test = nnf.BatchProvider(feat_provider_test)
  batch_provider_test.iter_per_batch = 50--opt.ipb
  batch_provider_test.nTimesMoreData = 10--opt.ntmd
  batch_provider_test.fg_fraction = opt.fg_frac
  batch_provider_test.bg_threshold = {0.0,0.5}
  batch_provider_test.do_flip = false
  batch_provider_test.batch_dim = feat_dim
  batch_provider_test:setupData()
  
  torch.save(testCache,batch_provider_test)
  feat_provider_test.model = features
end

--features = nil

--collectgarbage()
--collectgarbage()

--------------------------------------------------------------------------------
-- Compute conv5 feature cache (for SPP)
--------------------------------------------------------------------------------
if opt.algo == 'SPP' then
  print('Preparing conv5 features for '..ds_train.dataset_name..' '
        ..ds_train.image_set)
  local feat_cachedir = feat_provider.cachedir
  for i=1,ds_train:size() do
    xlua.progress(i,ds_train:size())
    local im_name = ds_train.img_ids[i]
    local cachefile = paths.concat(feat_cachedir,im_name)
    if not paths.filep(cachefile..'.h5') then
      local f = feat_provider:getConv5(i)
    end
    if not paths.filep(cachefile..'_flip.h5') then
      local f = feat_provider:getConv5(i,true)
    end
    if i%50 == 0 then
      collectgarbage()
      collectgarbage()
    end
  end
  
  print('Preparing conv5 features for '..ds_test.dataset_name..' '
        ..ds_test.image_set)
  local feat_cachedir = feat_provider_test.cachedir
  for i=1,ds_test:size() do
    xlua.progress(i,ds_test:size())
    local im_name = ds_test.img_ids[i]
    local cachefile = paths.concat(feat_cachedir,im_name)
    if not paths.filep(cachefile..'.h5') then
      local f = feat_provider_test:getConv5(i)
    end
    if i%50 == 0 then
      collectgarbage()
      collectgarbage()
    end
  end
end

--------------------------------------------------------------------------------
-- Prepare training model
--------------------------------------------------------------------------------

-- borrowed from https://github.com/soumith/imagenet-multiGPU.torch/blob/master/train.lua
-- clear the intermediate states in the model before saving to disk
-- this saves lots of disk space
local function sanitize(net)
  local list = net:listModules()
  for _,val in ipairs(list) do
    for name,field in pairs(val) do
      if torch.type(field) == 'cdata' then val[name] = nil end
      if name == 'homeGradBuffers' then val[name] = nil end
      if name == 'input_gpu' then val['input_gpu'] = {} end
      if name == 'gradOutput_gpu' then val['gradOutput_gpu'] = {} end
      if name == 'gradInput_gpu' then val['gradInput_gpu'] = {} end
      if (name == 'output' or name == 'gradInput') then
        val[name] = field.new()
      end
    end
  end
end

if opt.algo == 'RCNN' then
  classifier = model
end

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
  --sanitize(model)
  --torch.save(paths.concat(opt.save, 'model_' .. epoch .. '.t7'), classifier)
  --torch.save(paths.concat(opt.save, 'optimState_' .. epoch .. '.t7'), trainer.optimState)
end

sanitize(classifier)
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

