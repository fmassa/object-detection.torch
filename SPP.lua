require 'inn'
require 'mattorch'
require 'matio'

local SPP = torch.class('nnf.SPP')

--TODO vectorize code ?
function SPP:__init(dataset,model)

  self.dataset = dataset
  self.model = model
  self.spp_pooler = inn.SpatialPyramidPooling({{1,1},{2,2},{3,3},{6,6}})
  self.folder = ''
-- paper=864, their code=874 
  self.scales = {480,576,688,874,1200} -- 874
  self.randomscale = true
  self.sz_conv_standard = 13
  self.step_standard = 16
  self.offset0 = 21
  self.offset = 6.5
  
  self.inputArea = 224^2
  
  --self.cachedir = paths.concat(self.folder,self.model.name,self.dataset.dataset)
  --self.cachedir = '/media/francisco/rio/projects/pose_estimation/cachedir/conv5/Zeiler5/pascal/'
  self.cachedir = '/home/francisco/work/projects/pose_estimation/cachedir/conv5/Zeiler5/pascal/'
end

local function rgb2bgr(I)
  local out = I.new():resizeAs(I)
  for i=1,I:size(1) do
    out[i] = I[I:size(1)+1-i]
  end
  return out
end

local function prepareImage(I,typ)
  local typ = typ or 1
  local mean_pix = typ == 1 and {128.,128.,128.} or {103.939, 116.779, 123.68}
  local I = I
  if I:dim() == 2 then
    I = I:view(1,I:size(1),I:size(2))
  end
  if I:size(1) == 1 then
    I = I:expand(3,I:size(2),I:size(3))
  end
  I = rgb2bgr(I):mul(255)
  for i=1,3 do
    I[i]:add(-mean_pix[i])
  end
  return I
end

local function cleaningForward(input,model)
  local currentOutput = input
  for i=1,#model.modules do
          --print('jj '..i)
          --  --print(currentOutput:size())
    collectgarbage()
    currentOutput = model.modules[i]:updateOutput(currentOutput)
    model.modules[i].output = torch.Tensor():type(input:type())
    model.modules[i].gradInput = torch.Tensor():type(input:type())
    model.modules[i].gradWeight = torch.Tensor():type(input:type())
    model.modules[i].gradBias = torch.Tensor():type(input:type())
  end
  model.output = currentOutput
  return currentOutput
end

function SPP:getCrop(im_idx,bbox,flip)
  local flip = flip==nil and false or flip
  
  if self.curr_im_idx ~= im_idx or self.curr_doflip ~= flip then
    self.curr_im_idx = im_idx
    self.curr_im_feats = self:getConv5(im_idx,flip)
    self.curr_doflip = flip
  end
  
  local bbox = bbox
  --print(bbox)
  if flip then
    local tt = bbox[1]
    bbox[1] = self.curr_im_feats.imSize[3]-bbox[3]+1
    bbox[3] = self.curr_im_feats.imSize[3]-tt     +1
  end
  
  local bestScale,bestBbox = self:getBestSPPScale(bbox,self.curr_im_feats.imSize,self.curr_im_feats.scales)
  local box_norm = self:getResposeBoxes(bestBbox)

  local crop_feat = self:getCroppedFeat(self.curr_im_feats[bestScale],box_norm)
  
  return crop_feat  
end

function SPP:getFeature(im_idx,bbox,flip)
  local flip = flip==nil and false or flip
  
  local crop_feat = self:getCrop(im_idx,bbox,flip)
  
  local feat = self.spp_pooler:forward(crop_feat)
  
  return feat
end

function SPP:getConv5(im_idx,flip)
  local scales = self.scales
  local flip = flip==nil and false or flip
  --local imName = flip==true and imName..'_flip' or imName
  local to_matlab = to_matlab==nil and true or to_matlab
  local cachefile = paths.concat(self.cachedir,self.dataset.img_ids[im_idx])

  if flip then
    cachefile = cachefile..'_flip'
  end
  if paths.filep(cachefile) then
    local feats = torch.load(cachefile)
    return feats
  elseif paths.filep(cachefile..'.mat') then
    local f = mattorch.load(cachefile..'.mat')
    local feats = {}
    local idx = 1
    
    local lnames = {}
    for n in pairs(f) do
      table.insert(lnames,n)
    end
    table.sort(lnames)

    -- assert scales are consistent
    local read_scales = {}
    for i in pairs(f) do
      local scale=i:gsub('scale_','')
      table.insert(read_scales,tonumber(scale))
    end
    table.sort(read_scales)
    --assert(#scales == #read_scales, 'Number of scales do not match')
    for i=1,#scales do
      --assert(scales[i]==read_scales[i], 'Scales do not match')
    end

    for i,j in pairs(lnames) do
    if j == 'imSize' then
        feats.imSize = f[j]:squeeze()
      else
        feats[idx] = f[j]
        idx = idx+1
      end
    end
    
    feats.scales = read_scales
    
    return feats
  else
    local I = self.dataset:getImage(im_idx):float()
    I = prepareImage(I)
    if flip then
      I = image.hflip(I)
    end
    local rows = I:size(2)
    local cols = I:size(3)
    local feats = {}
--    model = model:float()
    for i=1,#scales do
--      local Ir = image.scale(I,'^'..scales[i])

      local sr = rows < cols and scales[i] or math.ceil(scales[i]*rows/cols)
      local sc = rows > cols and scales[i] or math.ceil(scales[i]*cols/rows)
      --local Ir = imresize(I,sc,sr):reshape(1,3,sr,sc):cuda()
      local Ir = image.scale(I,sc,sr):reshape(1,3,sr,sc):cuda()
      local f = self.model:forward(Ir):float():squeeze()
      --local f = cleaningForward(Ir,self.model):float():squeeze()
      if to_matlab then
        feats[string.format('scale_%.4d',scales[i])] = f
      else
        feats[i] = f
      end
      collectgarbage()
    end
    
    paths.mkdir(paths.dirname(cachefile))
    if to_matlab then
      feats.imSize = torch.FloatTensor({I:size(1),I:size(2),I:size(3)})
      mattorch.save(cachefile..'.mat',feats)
      return self:getConv5(im_idx,flip)
    else
      feats.imSize = I:size()
      torch.save(cachefile,feats)
    end
    
    return feats
  end
end

local function round(num)
  if num >= 0 then 
    return math.floor(num+.5)
  else 
    return math.ceil(num-.5)
  end
end

function SPP:getBestSPPScale(bbox,imSize,scales)

  local scales = scales or self.scales
  local min_dim = imSize[2]<imSize[3] and imSize[2] or imSize[3]
  
  local sz_conv_standard = self.sz_conv_standard
  local step_standard = self.step_standard

  local bestScale

  if self.randomscale then
    bestScale = torch.random(1,#scales)
  else
    local inputArea = self.inputArea
    local bboxArea = (bbox[4]-bbox[2]+1)*(bbox[3]-bbox[1]+1)
    
    local expected_scale = sz_conv_standard*step_standard*min_dim/math.sqrt(bboxArea)
    expected_scale = round(expected_scale)
    
    local nbboxDiffArea = torch.Tensor(#scales)

    for i=1,#scales do
      nbboxDiffArea[i] = math.abs(scales[i]-expected_scale)
    end
  
    _,bestScale = nbboxDiffArea:min(1)
    bestScale = bestScale[1]
  
  end

  local mul_factor = (scales[bestScale]-1)/(min_dim-1)

  local bestBbox = {((bbox[1]-1)*mul_factor+1),((bbox[2]-1)*mul_factor+1),
                    ((bbox[3]-1)*mul_factor+1),((bbox[4]-1)*mul_factor+1)}
                    
                    
  return bestScale,bestBbox
end

function SPP:getResposeBoxes(bbox)
  -- [x1 y1 x2 y2] order
  local offset0 = self.offset0
  local offset = self.offset
  local step_standard = self.step_standard
  
  local y0_norm = math.floor((bbox[2]-offset0 + offset)/step_standard + 0.5) + 1
  local x0_norm = math.floor((bbox[1]-offset0 + offset)/step_standard + 0.5) + 1
  
  local y1_norm = math.ceil((bbox[4] -offset0 - offset)/step_standard - 0.5) + 1
  local x1_norm = math.ceil((bbox[3] -offset0 - offset)/step_standard - 0.5) + 1
  
  if x0_norm > x1_norm then
    x0_norm = (x0_norm + x1_norm) / 2;
    x1_norm = x0_norm;
  end
  
  if y0_norm > y1_norm then
    y0_norm = (y0_norm + y1_norm) / 2;
    y1_norm = y0_norm;
  end
  
  local box_norm = {x0_norm,y0_norm,x1_norm,y1_norm}

  return box_norm
end

function SPP:getCroppedFeat(feat,bbox)
  -- [x1 y1 x2 y2] order
  
  local bbox_ = {}
  
  bbox_[2] = math.min(feat:size(2), math.max(1, bbox[2])); 
  bbox_[4] = math.min(feat:size(2), math.max(1, bbox[4])); 
  bbox_[1] = math.min(feat:size(3), math.max(1, bbox[1])); 
  bbox_[3] = math.min(feat:size(3), math.max(1, bbox[3]));  
  
  --local bbox = {bbox_[2],bbox_[1],bbox_[4],bbox_[3]}
  
  local patch = feat[{{},{bbox_[2],bbox_[4]},{bbox_[1],bbox_[3]}}]; -- attention car crop commence en 0 !

  return patch

end

function SPP:float()
  self.spp_pooler = self.spp_pooler:float()
  return self
end
