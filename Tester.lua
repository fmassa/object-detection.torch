local objdet = require 'objdet.env'
local utils = require('objdet.utils')
local nms = require('objdet.nms')

local keep_top_k = utils.keep_top_k
local VOCevaldet = utils.VOCevaldet

local Tester = torch.class('objdet.Tester',objdet)

function Tester:__init(module,feat_provider,dataset)
  self.dataset = dataset
  self.feat_provider = feat_provider
  self.module = module

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

local function print_scores(dataset,res)
  local str = {}
  table.insert(str,'Results:\n')
  -- print class names
  table.insert(str, '|')
  for i = 1, dataset.num_classes do
    table.insert(str, ('%5s|'):format(dataset.classes[i]))
  end
  table.insert(str, '\n|')
  -- print class scores
  for i = 1, dataset.num_classes do
    local l = #dataset.classes[i] < 5 and 5 or #dataset.classes[i]
    local l = res[i] == res[i] and l-5 or l-3
    if l > 0 then
      table.insert(str, ('%.3f%'..l..'s|'):format(res[i],' '))
    else
      table.insert(str, ('%.3f|'):format(res[i]))
    end
  end
  table.insert(str,('\n'))
  table.insert(str,('Avg.: %.4f\n'):format(res:mean(1)[1]))
  return table.concat(str,'')
end

function Tester:pruneDetections(aboxes, thresh, output, boxes, i, max_per_image, max_per_set)
  local add_bg = 1
  self._scored_boxes = self._scored_boxes or torch.FloatTensor()
  local scored_boxes = self._scored_boxes
  local dataset = self.dataset
  -- do a NMS for each class, based on the scores from the classifier
  for j=1,dataset.num_classes do
    local scores = output:select(2,j+add_bg)
    -- only select detections with a score greater than thresh
    -- this avoid doing NMS on too many bboxes with low score
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

    -- remove low scoring boxes and update threshold
    if i%1000 == 0 then
      aboxes[j],thresh[j] = keep_top_k(aboxes[j],max_per_set)
    end

  end
end

function Tester:finalPrunning(aboxes, thresh)
  local dataset = self.dataset
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
  end
end

function Tester:evaluate(aboxes)
  local dataset = self.dataset
  local res = {}
  for i=1,dataset.num_classes do
    local cls = dataset.classes[i]
    res[i] = VOCevaldet(dataset,aboxes[i],cls)
  end
  res = torch.Tensor(res)
  return res
end

function Tester:test(iteration)
  
  local dataset = self.dataset
  local module = self.module
  local feat_provider = self.feat_provider

  module:evaluate()
  feat_provider:evaluate()
  dataset:loadROIDB()
  
  local detec = objdet.ImageDetect(module, feat_provider)
  local boxes
  local im
  local output

  local aboxes = {}
  for i=1,dataset.num_classes do
    table.insert(aboxes,{})
  end

  local num_images = dataset:size()
  
  local max_per_set = 5*num_images
  local max_per_image = 100
  local thresh = torch.ones(dataset.num_classes):mul(0.05)
  
  local timer = torch.Timer()
  local timer_l = torch.Timer()

  -- SPP is more efficient if we cache the features. We treat it differently then
  -- the other feature providers
  local pass_index = torch.type(feat_provider) == 'objdet.SPP' and true or false

  local display_string =
      'test: (%s) %5d/%-5d forward time: %.3f, select time: %.3fs, total time: %.3fs'
  for i = 1, num_images do
    timer:reset()

    if pass_index then
      im = i
    else
      im = dataset:getImage(i)
    end
    boxes = dataset:getROIBoxes(i):float()

    timer_l:reset()
    output,boxes = detec:detect(im,boxes)

    local detect_time = timer_l:time().real
    
    timer_l:reset()
    self:pruneDetections(aboxes, thresh, output, boxes, i, max_per_image, max_per_set)

    local total_time = timer:time().real
    local prune_time = timer_l:time().real

    print(display_string:format(dataset.dataset_name, i, num_images,
                                detect_time, prune_time, total_time))
  end

  local pathfolder = paths.concat(self.cachefolder,'test_iter'..iteration)
  paths.mkdir(pathfolder)

  self:finalPrunning(aboxes, thresh)

  save_file = paths.concat(pathfolder, 'boxes_'..
                           dataset.dataset_name..self.suffix)
  torch.save(save_file, aboxes)

  local res = self:evaluate(aboxes)

  if torch.isTensor(res) then
    print(print_scores(dataset,res))
  else
    for k, v in pairs(res) do
      print(k)
      print(print_scores(dataset,v))
    end
  end

  -- clean roidb to free memory
  dataset.roidb = nil
  return res
end
