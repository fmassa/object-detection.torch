require 'xlua'
--require 'mattorch'

local Tester = torch.class('nnf.Tester')

function Tester:__init(module,feat_provider)
  self.dataset = feat_provider.dataset
  self.module = module
  self.feat_provider = feat_provider
  --self.batch_provider

  self.feat_dim = {256*50}
  self.batch_size = 128
  self.max_batch_size = 4000
  
  self.cachefolder = nil
  self.cachename = nil
  self.verbose = true
end

-- improve it !
function Tester:validate(criterion)

  local tname = paths.concat(self.cachefolder,self.cachename)
  local valData
  if paths.filep(tname) then
    valData = torch.load(tname)
  else
    -- batch_provider need to be set before
    valData = {}
    valData.inputs,valData.targets = self.batch_provider:getBatch()
    torch.save(tname,valData)
    self.batch_provider = nil
  end

  local num_batches = valData.inputs:size(1)
  local module = self.module

  local err = 0
  local inputs = torch.CudaTensor()
  local targets = torch.CudaTensor()
  for t=1,num_batches do
    xlua.progress(t,num_batches)
    
    inputs:resize(valData.inputs[t]:size()):copy(valData.inputs[t])
    targets:resize(valData.targets[t]:size()):copy(valData.targets[t])
    
    local output = module:forward(inputs)
    
    err = err + criterion:forward(output,targets)
  end
  
  valData = nil
  collectgarbage()
  
  return err/num_batches
end

function Tester:test(iteration)
  
  local dataset = self.dataset
  local module = self.module
  local feat_provider = self.feat_provider

  local pathfolder = paths.concat(self.cachefolder,'test_iter'..iteration)
  paths.mkdir(pathfolder)  

  module:evaluate()
  dataset:loadROIDB()
  
  local feats = torch.FloatTensor()
  local feats_batched = {}
  local feats_cuda = torch.CudaTensor()
  
  local output = torch.FloatTensor()
  
  local output_dim = module:get(module:size())
  
  local boxes
  
  for i=1,dataset:size() do
    xlua.progress(i,dataset:size())
    boxes = dataset.roidb[i]
    local num_boxes = boxes:size(1)

    feats:resize(num_boxes,unpack(self.feat_dim))
    for idx=1,num_boxes do
      feats[idx] = feat_provider:getFeature(i,boxes[idx])
    end
    torch.split(feats_batched,self.max_batch_size,1)
    
    for idx,f in ipairs(feats_batched) do
      local fs = f:size(1)
      feats_cuda:resize(fs,unpack(self.feat_dim)):copy(f)
      module:forward(feats_cuda)
      if idx == 1 then
        local out_size = module.output:size()
        output:resize(num_boxes,unpack(out_size[{{2,-1}}]:totable()))
      end
      output:narrow(1,(idx-1)*self.max_batch_size+1,fs):copy(module.output)
    end

    --collectgarbage()
    --mattorch.save(paths.concat(pathfolder,dataset.img_ids[i]..'.mat'),output:double())
  end
  
  -- clean roidb to free memory
  dataset.roidb = nil
end
