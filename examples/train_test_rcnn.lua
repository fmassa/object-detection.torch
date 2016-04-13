require 'nnf'

cmd = torch.CmdLine()
cmd:text('Example on how to train/test a RCNN based object detector on Pascal')
cmd:text('')
cmd:text('Options:')
cmd:option('-name',      'rcnn-example', 'base name')
cmd:option('-modelpath', '',             'path to the pre-trained model')
cmd:option('-lr',        1e-3,           'learning rate')
cmd:option('-num_iter',  40000,          'number of iterations')
cmd:option('-disp_iter', 100,            'display every n iterations')
cmd:option('-lr_step',   30000,          'step for reducing the learning rate')
cmd:option('-save_step', 10000,          'step for saving the model')
cmd:option('-gpu',       1,              'gpu to use (0 for cpu mode)')
cmd:option('-seed',      1,              'fix random seed (if ~= 0)')
cmd:option('-numthreads',6,              'number of threads')

opt = cmd:parse(arg or {})

assert(paths.filep(opt.modelpath), 'need to provide the path for the pre-trained model')

exp_name = cmd:string(opt.name, opt, {name=true, gpu=true, numthreads=true,
                                      modelpath=true})

rundir = '../cachedir/'..exp_name
paths.mkdir(rundir)

cmd:log(paths.concat(rundir,'log'), opt)
cmd:addTime('RCNN Example')

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

if opt.seed ~= 0 then
  torch.manualSeed(opt.seed)
  if opt.gpu > 0 then
    cutorch.manualSeed(opt.seed)
  end
  print('Using fixed seed: '..opt.seed)
end

torch.setnumthreads(opt.numthreads)

--------------------------------------------------------------------------------
-- define model and criterion
--------------------------------------------------------------------------------
-- load pre-trained model for finetuning
-- should already have the right number of outputs in the last layer,
-- which can be done by removing the last layer and replacing it by a new one
-- for example:
-- pre_trained_model:remove() -- remove last layer
-- pre_trained_model:add(nn.Linear(4096,21)) -- add new layer
model = torch.load(opt.modelpath)

criterion = nn.CrossEntropyCriterion()

model:type(tensor_type)
criterion:type(tensor_type)

print('Model:')
print(model)
print('Criterion:')
print(criterion)

-- define the transformations to do in the image before
-- passing it to the network
local image_transformer= nnf.ImageTransformer{
  mean_pix={102.9801,115.9465,122.7717},
  raw_scale = 255,
  swap = {3,2,1}
}

print(image_transformer)
--------------------------------------------------------------------------------
-- define data for training
--------------------------------------------------------------------------------

-- this class holds all the necessary informationn regarding the dataset
ds = nnf.DataSetPascal{
  image_set='trainval',
  datadir='datasets/VOCdevkit',
  roidbdir='data/selective_search_data',
  year=2007
}
print('DataSet Training:')
print(ds)
--------------------------------------------------------------------------------
-- define feature providers
--------------------------------------------------------------------------------

local crop_size = 224

-- the feature provider extract the features for a given image + bounding box
fp = nnf.RCNN{
  image_transformer=image_transformer,
  crop_size=crop_size,
  num_threads=opt.numthreads
}
-- different frameworks can behave differently during training and testing
fp:training()

print('Feature Provider:')
print(fp)

--------------------------------------------------------------------------------
-- define batch providers
--------------------------------------------------------------------------------

bp = nnf.BatchProviderRC{
  dataset=ds,
  feat_provider=fp,
  bg_threshold={0.0,0.5},
  nTimesMoreData=2,
  iter_per_batch=10,--100,
}
bp:setupData()

print('Batch Provider:')
print(bp)
--------------------------------------------------------------------------------
-- train
--------------------------------------------------------------------------------

trainer = nnf.Trainer(model, criterion, bp)

local num_iter = opt.num_iter/opt.disp_iter
local lr_step = opt.lr_step/opt.disp_iter
local save_step = opt.save_step/opt.disp_iter

trainer.optimState.learningRate = opt.lr

local lightModel = model:clone('weight','bias')

-- main training loop
for i=1,num_iter do
  if i % lr_step == 0 then
    trainer.optimState.learningRate = trainer.optimState.learningRate/10
  end
  print(('Iteration %3d/%-3d'):format(i,num_iter))
  trainer:train(opt.disp_iter)
  print(('  Training error: %.5f'):format(trainer.fx[i]))

  if i% save_step == 0 then
    torch.save(paths.concat(rundir, 'model.t7'), lightModel)
  end
end

torch.save(paths.concat(rundir, 'model.t7'), lightModel)

--------------------------------------------------------------------------------
-- evaluation
--------------------------------------------------------------------------------
-- add softmax to classifier, because we were using nn.CrossEntropyCriterion
local softmax = nn.SoftMax()
softmax:type(tensor_type)
model:add(softmax)

-- dataset for evaluation
dsv = nnf.DataSetPascal{
  image_set='test',
  datadir='datasets/VOCdevkit',
  roidbdir='data/selective_search_data',
  year=2007
}
print('DataSet Evaluation:')
print(dsv)

-- feature provider for evaluation
fpv = nnf.RCNN{
  image_transformer=image_transformer,
  crop_size=crop_size,
  num_threads=opt.numthreads
}
fpv:evaluate()
print('Feature Provider Evaluation:')
print(fpv)

-- define the class to test the model on the full dataset
tester = nnf.Tester(model, fpv, dsv)
tester.cachefolder = rundir
tester:test(opt.num_iter)
