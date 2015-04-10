require 'xlua'
require 'mattorch'

local Tester = torch.class('nnf.Tester')

function Tester:__init(module,feat_provider)
  self.dataset = feat_provider.dataset
  self.module = module
  self.feat_provider = feat_provider
  --self.batch_provider

  self.feat_dim = {256*50}
  self.batch_size = 128
  self.max_batch_size = 15000
  
  self.cachefolder = 'results_regression'
  self.cachename = 'pascal3d_val_ds2.data'
  self.verbose = true
end

-- improve it !
function Tester:validate(criterion)

  local tname = paths.concat(self.cachefolder,self.cachename)
  local valData
  if paths.filep(tname) then
    valData = torch.load(tname)
  else
    -- batch_provider need to be set
    valData = {}
    valData.batches,valData.targets = self.batch_provider:getBatch()
    torch.save(tname,valData)
    self.batch_provider = nil
  end

  --local valData = torch.load('pascal3d_val_ds2.data')
  local batch_size = valData.batches:size(2)
  local num_batches = valData.batches:size(1)
  
  local module = self.module
  
  --local output_size = module:get(module:size()).weight:size(1)
  
  --local output = torch.Tensor(num_batches,batch_size,output_size)
  local err = 0
  for t=1,num_batches do
    xlua.progress(t,num_batches)
    
    local inputs = valData.batches[t]:cuda()
    local targets = valData.targets[t]:float():cuda()
    
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
  --local batch_size = self.batch_size

  local pathfolder = paths.concat(self.cachefolder,'test_iter'..iteration)
  paths.mkdir(pathfolder)  

  module:evaluate()
  dataset:loadROIDB()
  
  local feats = torch.Tensor():float()
  local output = torch.Tensor():cuda()--float()
  
  local boxes
  
  for i=1,dataset:size() do
    xlua.progress(i,dataset:size())
    boxes = dataset.roidb[i]
    local num_boxes = boxes:size(1)
    
    --local batch_size = num_boxes > self.max_batch_size and self.batch_size or num_boxes
    --local num_batches = math.ceil(num_boxes/batch_size)
    --local batch_rest = num_boxes%batch_size
    
    --feats:resize(batch_size,unpack(feat_dim))
    feats:resize(num_boxes,unpack(self.feat_dim))
    for idx=1,num_boxes do
      feats[idx] = feat_provider:getFeature(i,boxes[idx])
    end
    output = module:forward(feats:cuda())
    
    --[[ make more general later, not in the mood
    for b = 1,num_batches-1 do
    
      for idx=1,batch_size do
        feats[idx] = feat_provider:getFeature(i,boxes[(b-1)*batch_size + idx])
      end
      
      output = module:forward(feats)
      
    end]]
    collectgarbage()
    --torch.save(paths.concat(self.cachefolder,module.experiment,))
    mattorch.save(paths.concat(pathfolder,dataset.img_ids[i]..'.mat'),output:double())
  end
  
  -- clean roidb to free memory
  dataset.roidb = nil
end
