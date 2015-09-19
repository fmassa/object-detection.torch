local utils = paths.dofile('utils.lua')
local nms = paths.dofile('nms.lua')

local keep_top_k = utils.keep_top_k
local VOCevaldet = utils.VOCevaldet

local Tester = torch.class('nnf.Tester_FRCNN')

function Tester:__init(module,feat_provider)
  self.dataset = feat_provider.dataset
  self.module = module
  self.feat_provider = feat_provider

  self.feat_dim = {256*50}
  self.max_batch_size = 4000
  
  self.cachefolder = nil
  self.cachename = nil
  self.suffix = ''
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

  module:evaluate()
  feat_provider:evaluate()
  dataset:loadROIDB()
  
  local detec = nnf.ImageDetect(module, feat_provider)
  local boxes
  local im

  local aboxes = {}
  for i=1,dataset.num_classes do
    table.insert(aboxes,{})
  end
  
  local max_per_set = 5*dataset:size()
  local max_per_image = 100
  local thresh = torch.ones(dataset.num_classes):mul(-1.5)
  local scored_boxes = torch.FloatTensor()
  
  local timer = torch.Timer()
  local timer2 = torch.Timer()
  local timer3 = torch.Timer()

  for i=1,dataset:size() do
    timer:reset()
    io.write(('test: (%s) %5d/%-5d '):format(dataset.dataset_name,i,dataset:size()));
    boxes = dataset:getROIBoxes(i):float()
    im = dataset:getImage(i)
    timer3:reset()
    local output = detec:detect(im,boxes)

    local add_bg = 1--0
    --if dataset.num_classes ~= output:size(2) then -- if there is no svm
      --output = softmax:forward(output) 
    --  add_bg = 1
    --end
    local tt = 0 
    local tt2 = timer3:time().real
    
    timer2:reset()
    for j=1,dataset.num_classes do
      local scores = output:select(2,j+add_bg)
      local idx = torch.range(1,scores:numel()):long()
      local idx2 = scores:gt(thresh[j])
      idx = idx[idx2]
      scored_boxes:resize(idx:numel(),5)
      if scored_boxes:numel() > 0 then
        scored_boxes:narrow(2,1,4):index(boxes,1,idx)
        scored_boxes:select(2,5):copy(scores[idx2])
      end
      local keep = nms(scored_boxes,0.3)
      if keep:numel()>0 then
        local _,ord = torch.sort(scored_boxes:select(2,5):index(1,keep),true)
        ord = ord:narrow(1,1,math.min(ord:numel(),max_per_image))
        keep = keep:index(1,ord)
        aboxes[j][i] = scored_boxes:index(1,keep)
      else
        aboxes[j][i] = torch.FloatTensor()
      end
      
      if i%1000 == 0 then
        aboxes[j],thresh[j] = keep_top_k(aboxes[j],max_per_set)
      end
      
    end

    io.write((' prepare feat time: %.3f, forward time: %.3f, select time: %.3fs, total time: %.3fs\n'):format(tt,tt2,timer2:time().real,timer:time().real));
  end

  local pathfolder = paths.concat(self.cachefolder,'test_iter'..iteration)
  paths.mkdir(pathfolder)

  for i = 1,dataset.num_classes do
    -- go back through and prune out detections below the found threshold
    for j = 1,dataset:size() do
      if aboxes[i][j]:numel() > 0 then
        local I = aboxes[i][j]:select(2,5):lt(thresh[i])
        local idx = torch.range(1,aboxes[i][j]:size(1)):long()
        idx = idx[I]
        if idx:numel()>0 then
          aboxes[i][j] = aboxes[i][j]:index(1,idx)
        end
      end
    end
    --save_file = paths.concat(pathfolder, dataset.classes[i].. '_boxes_'..
    --                         dataset.dataset_name..self.suffix)
    --torch.save(save_file, aboxes)
  end
  save_file = paths.concat(pathfolder, 'boxes_'..
                           dataset.dataset_name..self.suffix)
  torch.save(save_file, aboxes)


  local res = {}
  for i=1,dataset.num_classes do
    local cls = dataset.classes[i]
    res[i] = VOCevaldet(dataset,aboxes[i],cls)
  end
  res = torch.Tensor(res)
  print('Results:')
  -- print class names
  io.write('|')
  for i = 1, dataset.num_classes do
    io.write(('%5s|'):format(dataset.classes[i]))
  end
  io.write('\n|')
  -- print class scores
  for i = 1, dataset.num_classes do
    local l = #dataset.classes[i] < 5 and 5 or #dataset.classes[i]
    local l = res[i] == res[i] and l-5 or l-3
    if l > 0 then
      io.write(('%.3f%'..l..'s|'):format(res[i],' '))
    else
      io.write(('%.3f|'):format(res[i]))
    end
  end
  io.write('\n')
  io.write(('mAP: %.4f\n'):format(res:mean(1)[1]))

  -- clean roidb to free memory
  dataset.roidb = nil
  return res
end
