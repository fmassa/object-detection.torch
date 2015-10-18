require 'nnf'

cmd = torch.CmdLine()
cmd:text('Example on how to train/test a RCNN based object detector on Pascal')
cmd:text('')
cmd:text('Options:')
cmd:option('-name',      'rcnn-example', 'base name')
cmd:option('-lr',        1e-3,           'learning rate')
cmd:option('-num_iter',  40000,          'number of iterations')
cmd:option('-disp_iter', 100,            'display every n iterations')
cmd:option('-lr_step',   30000,          'step for reducing the learning rate')
cmd:option('-gpu',       1,              'gpu to use (0 for cpu mode)')

opt = cmd:parse(arg or {})

exp_name = cmd:string(opt.name, opt, {name=true, gpu=true})

rundir = '../cachedir/'..exp_name
paths.mkdir(rundir)

local tensor_type
if opt.gpu > 0 then
  require 'cunn'
  cutorch.setDevice(opt.gpu)
  tensor_type = 'torch.CudaTensor'
else
  require 'nn'
  tensor_type = 'torch.FloatTensor'
end

--------------------------------------------------------------------------------
-- define data
--------------------------------------------------------------------------------

-- this class holds all the necessary informationn regarding the dataset
ds = nnf.DataSetPascal{
  image_set='trainval',
  datadir='datasets/VOCdevkit',
  roidbdir='data/selective_search_data'
}
-- define the transformations to do in the image before
-- passing it to the network
local image_transformer= nnf.ImageTransformer{
  mean_pix={102.9801,115.9465,122.7717},
  raw_scale = 255,
  swap = {3,2,1}
}
--------------------------------------------------------------------------------
-- define feature providers
--------------------------------------------------------------------------------

local crop_size = 224

-- the feature provider extract the features for a given image + bounding box
fp = nnf.RCNN{
  image_transformer=image_transformer,
  crop_size=crop_size
}
-- different frameworks can behave differently during training and testing
fp:training()
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

--------------------------------------------------------------------------------
-- define model and criterion
--------------------------------------------------------------------------------
paths.dofile('../models/rcnn.lua')
model = createModel()

criterion = nn.CrossEntropyCriterion()

model:type(tensor_type)
criterion:type(tensor_type)
--------------------------------------------------------------------------------
-- train
--------------------------------------------------------------------------------

trainer = nnf.Trainer(model, criterion, bp)

local num_iter = opt.num_iter/opt.disp_iter
local step_iter = opt.lr_step/opt.disp_iter

trainer.optimState.learningRate = opt.lr

local lightModel = model:clone('weight','bias')

-- main training loop
for i=1,num_iter do
  if i % lr_step == 0 then
    trainer.optimState.learningRate = trainer.optimState.learningRate/10
  end
  print(('Iteration %3d/%-3d'):format(i,num_iter))
  trainer:train(opt.disp_iter)

  if i% save_step == 0 then
    torch.save(paths.concat(rundir, 'model.t7'), lightModel)
  end
end

torch.save(paths.concat(rundir, 'model.t7'), lightModel)
--------------------------------------------------------------------------------
-- evaluate
--------------------------------------------------------------------------------

-- add softmax to classfier, because we were using nn.CrossEntropyCriterion
local softmax = nn.SoftMax()
softmax:type(tensor_type)
model:add(softmax)

-- dataset for evaluation
dsv = nnf.DataSetPascal{
  image_set='test',
  datadir='datasets/VOCdevkit',
  roidbdir='data/selective_search_data'
}

-- feature provider for evaluation
fpv = nnf.RCNN{
  image_transformer=image_transformer,
  crop_size=crop_size
}
fpv:evaluate()

-- define the class to test the model on the full dataset
tester = nnf.Tester(model, fpv, dsv)
tester.cachefolder = rundir
tester:test(opt.num_iter)
