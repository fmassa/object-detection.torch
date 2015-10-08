require 'nnf'
require 'inn'
require 'cudnn'
require 'gnuplot'

cutorch.setDevice(2)

dt = torch.load('pascal_2007_train.t7')
if false then
  ds = nnf.DataSetPascal{image_set='train',
                         datadir='/home/francisco/work/datasets/VOCdevkit',
                         roidbdir='/home/francisco/work/datasets/rcnn/selective_search_data'
                        }
else
  ds = nnf.DataSetPascal{image_set='trainval',
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
elseif false then
  ds.roidb = dt.roidb
end

local image_transformer= nnf.ImageTransformer{mean_pix={102.9801,115.9465,122.7717},--{103.939, 116.779, 123.68},
                                              raw_scale = 255,
                                              swap = {3,2,1}}
if true then
  bp = nnf.BatchProviderROI(ds)
  bp.image_transformer = image_transformer
  bp.bg_threshold = {0.1,0.5}
  bp:setupData()
else
  bp = nnf.BatchProviderROI(ds)
  bp.image_transformer = image_transformer
  local temp = torch.load('pascal_2007_train_bp.t7')
  bp.bboxes = temp.bboxes
end


if false then
  local mytest = nnf.ROIPooling(50,50):float()
  function do_mytest()
    local input0,target0 = bp:getBatch(input0,target0)
    local o = mytest:forward(input0)
    return input0,target0,o
  end
  --input0,target0,o = do_mytest()
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
  
  elseif false then
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

  else
    local fold = 'data/models/imagenet_models/alexnet/'
    local m1 = torch.load(fold..'features.t7')
    local m2 = torch.load(fold..'top.t7')

    for i=1,14 do
      features:add(m1:get(i):clone())
    end
    features:get(3).padW = 1
    features:get(3).padH = 1
    features:get(7).padW = 1
    features:get(7).padH = 1
 
    for i=2,7 do
      classifier:add(m2:get(i):clone())
    end
    local linear = nn.Linear(4096,21):cuda()
    linear.weight:normal(0,0.01)
    linear.bias:zero()
    classifier:add(linear)
  end
  collectgarbage()

  local prl = nn.ParallelTable()
  prl:add(features)
  prl:add(nn.Identity())
  model:add(prl)
  --model:add(nnf.ROIPooling(6,6):setSpatialScale(1/16))
  model:add(inn.ROIPooling(6,6):setSpatialScale(1/16))
  model:add(nn.View(-1):setNumInputDims(3))
  model:add(classifier)

end
print(model)

model:cuda()
parameters,gradParameters = model:getParameters()

parameters2,gradParameters2 = model:parameters()

lr = {0,0,1,2,1,2,1,2,1,2,1,2,1,2,1,2}
wd = {0,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0}

local function updateGPlrwd(clr)
  local clr = clr or 1
  for i,p in pairs(gradParameters2) do
    p:add(wd[i]*0.0005,parameters2[i])
    p:mul(lr[i]*clr)
  end
end

optimState = {learningRate = 1,--1e-3,
              weightDecay = 0.000, momentum = 0.9,
              learningRateDecay = 0, dampening=0}

--------------------------------------------------------------------------
-- training
--------------------------------------------------------------------------

confusion_matrix = optim.ConfusionMatrix(21)


model:training()

savedModel = model:clone('weight','bias','running_mean','running_std')

criterion = nn.CrossEntropyCriterion():cuda()
--criterion.nll.sizeAverage = false

--normalize = true

display_iter = 20

--inputs = {torch.CudaTensor(),torch.FloatTensor()}
inputs = {torch.CudaTensor(),torch.CudaTensor()}
target = torch.CudaTensor()

learningRate = 1e-3

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

      -- mimic different learning rates per layer
      -- without the cost of having a huge tensor
      updateGPlrwd(learningRate)

      if normalize then
        gradParameters:div(batchSize)
        f = f/batchSize
      end
      
      confusion_matrix:batchAdd(outputs,target)

      return f,gradParameters
    end

    local x,fx = optim.sgd(feval,parameters,optimState)
    err = err + fx[1]
  end
  print('Training error: '..err/display_iter)
  return err/display_iter
end

epoch_size = math.ceil(ds:size()/bp.imgs_per_batch)
stepsize = 30000--30000
print_step = 10
num_iter = 40000--40000
num_iter = num_iter/display_iter--3000

confusion_matrix:zero()
train_err = {}
exp_name = 'frcnn_t11'

paths.mkdir(paths.concat('cachedir',exp_name))
--logger = optim.Logger(paths.concat('cachedir',exp_name,'train_err.log'))
train_acc = {}
for i=1,num_iter do

  if i%(stepsize/display_iter) == 0 then
    --optimState.learningRate = optimState.learningRate/10
    learningRate = learningRate/10
  end
  
  --print(('Iteration: %d/%d, lr: %.5f'):format(i,num_iter,optimState.learningRate))
  print(('Iteration: %d/%d, lr: %.5f'):format(i,num_iter,learningRate))

  local t_err = train()
  table.insert(train_err,t_err)


  if i%print_step == 0 then
    print(confusion_matrix)
    table.insert(train_acc,confusion_matrix.averageUnionValid*100)
    gnuplot.epsfigure(paths.concat('cachedir',exp_name,'train_err.eps'))
    gnuplot.plot('train',torch.Tensor(train_acc),'-')
    gnuplot.xlabel('Iterations (200 batch update)')
    gnuplot.ylabel('Training accuracy')
    gnuplot.grid('on')
    gnuplot.plotflush()
    gnuplot.closeall()

    confusion_matrix:zero()
  end

  if i%100 == 0 then
    torch.save(paths.concat('cachedir',exp_name..'.t7'),savedModel)
  end
end

-- test
dsv = nnf.DataSetPascal{image_set='test',
                         datadir='datasets/VOCdevkit',
                         roidbdir='data/selective_search_data'
                         }


local fpv = {dataset=dsv}
tester = nnf.Tester_FRCNN(model,fpv)
tester.cachefolder = 'cachedir/'..exp_name
tester:test(num_iter)
