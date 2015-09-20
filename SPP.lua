local hdf5 = require 'hdf5'
local flipBoundingBoxes = paths.dofile('utils.lua').flipBoundingBoxes

local SPP = torch.class('nnf.SPP')

-- argcheck crashes with that many arguments, and using unordered
-- doesn't seems practical
--[[
local argcheck = require 'argcheck'
local initcheck = argcheck{
  pack=true,
  {name="model",
   type="nn.Sequential",
   help="conv5 model"},
  {name="dataset",
   type="nnf.DataSetPascal", -- change to allow other datasets
   opt=true,
   help="A dataset class"},
  {name="pooling_scales",
   type="table",
   default={{1,1},{2,2},{3,3},{6,6}},
   help="pooling scales"},
  {name="num_feat_chns",
   type="number",
   default=256,
   help="number of feature channels to be pooled"},
  {name="scales",
   type="table",
   default={480,576,688,874,1200},
   help="image scales"},
  {name="sz_conv_standard",
   type="number",
   default=13,
   help=""},
  {name="step_standard",
   type="number",
   default=16,
   help=""},
  {name="offset0",
   type="number",
   default=21,
   help=""},
  {name="offset",
   type="number",
   default=6.5,
   help=""},
  {name="inputArea",
   type="number",
   default=224^2,
   help="force square crops"},
  {name="image_transformer",
   type="nnf.ImageTransformer",
   default=nnf.ImageTransformer{},
   help="Class to preprocess input images"},
  {name="use_cache",
   type="boolean",
   default=true,
   help=""},
  {name="cachedir",
   type="string",
   opt=true,
   help=""},
}
--]]


function SPP:__init(...)

  self.dataset = dataset
  self.model = model

  --local opts = initcheck(...)
  --for k,v in pairs(opts) do self[k] = v end

  self.num_feat_chns = 256
  self.pooling_scales = {{1,1},{2,2},{3,3},{6,6}}
  local pyr = torch.Tensor(self.pooling_scales):t()
  local pooled_size = pyr[1]:dot(pyr[2])
  self.output_size = {self.num_feat_chns*pooled_size}

  --self.spp_pooler = inn.SpatialPyramidPooling(self.pooling_scales):float()
  self.image_transformer = nnf.ImageTransformer{}
-- [[
-- paper=864, their code=874 
  self.scales = {480,576,688,874,1200} -- 874
  
  self.sz_conv_standard = 13
  self.step_standard = 16
  self.offset0 = 21
  self.offset = 6.5
  
  self.inputArea = 224^2
  
  self.use_cache = true

  self.cachedir = nil
  --]]
  self.train = true
end

function SPP:training()
  self.train = true
end

function SPP:evaluate()
  self.train = false
end

-- here just to check
function SPP:getCrop_old(im_idx,bbox,flip)
  local flip = flip or false
  
  if self.curr_im_idx ~= im_idx or self.curr_doflip ~= flip then
    self.curr_im_idx = im_idx
    self.curr_im_feats = self:getConv5(im_idx,flip)
    self.curr_doflip = flip
  end

  if flip then
    flipBoundingBoxes(bbox,self.curr_im_feats.imSize[3])
  end
  
  local bestScale,bestBbox = self:getBestSPPScale(bbox,self.curr_im_feats.imSize,self.curr_im_feats.scales)
  local box_norm = self:getResposeBoxes(bestBbox)

  local crop_feat = self:getCroppedFeat(self.curr_im_feats.rsp[bestScale],box_norm)

  return crop_feat
end

function SPP:getCrop(im_idx,bbox,flip)
  local flip = flip or false
  
  if self.curr_im_idx ~= im_idx or self.curr_doflip ~= flip then
    self.curr_im_idx = im_idx
    self.curr_im_feats = self:getConv5(im_idx,flip)
    self.curr_doflip = flip
  end

  if type(bbox) == 'table' then
    bbox = torch.FloatTensor(bbox)
  end
  bbox = bbox:dim() == 1 and bbox:view(1,-1) or bbox

  if flip then
    flipBoundingBoxes(bbox,self.curr_im_feats.imSize[3])
  end
  
  local feat = self.curr_im_feats
  local bestScale,bestbboxes,bboxes_norm,projected_bb =
            self:projectBoxes(feat, bbox, feat.scales)

  local crop_feat = {}
  for i=1,bbox:size(1) do
    local bbox_ = projected_bb[i]
    local patch = feat.rsp[bestScale[i]][{{},{bbox_[2],bbox_[4]},{bbox_[1],bbox_[3]}}]
    table.insert(crop_feat,patch)
  end
  
  return crop_feat  
end

-- here just to check
function SPP:getFeature_old(im_idx,bbox,flip)
  local flip = flip or false

  local crop_feat = self:getCrop_old(im_idx,bbox,flip)

  local feat = self.spp_pooler:forward(crop_feat)
  return feat
end


function SPP:getFeature(im_idx,bbox,flip)
  local flip = flip or false

  local crop_feat = self:getCrop(im_idx,bbox,flip)

  self._feat = self._feat or torch.FloatTensor()
  self._feat:resize(#crop_feat,table.unpack(self.output_size))
  for i=1,#crop_feat do
    self._feat[i]:copy(self.spp_pooler:forward(crop_feat[i]))
  end

  return self._feat
end

-- SPP is meant to keep a cache of the conv5 features
-- for fast training. In this case, we suppose that
-- we provide the image index in the dataset.
-- We can also use an image as input, in which case it
-- won't save a conv5 cache.
function SPP:getConv5(im_idx,flip)
  local scales = self.scales
  local flip = flip or false
  local cachedir = self.cachedir
  
  assert(cachedir or (not self.use_cache), 
         'Need to set a folder to save the conv5 features')
  
  if not cachedir then
    cachedir = ''
  end

  local im_name
  if not self.dataset then
    self.use_cache = false
    im_name = ''
  else
    im_name = self.dataset.img_ids[im_idx]
  end
  
  local cachefile = paths.concat(cachedir,im_name)

  if flip then
    cachefile = cachefile..'_flip'
  end
  local feats
  if self.use_cache and paths.filep(cachefile..'.h5') then
    local f = hdf5.open(cachefile..'.h5','r')
    feats = f:read('/'):all()
    f:close()
    feats.scales = feats.scales:totable()
    for i=1,#feats.scales do
      feats.rsp[i] = feats.rsp[tostring(i)]
      feats.rsp[tostring(i)] = nil
    end
  else
    local I
    if type(im_idx) == 'number' and self.dataset then
      I = self.dataset:getImage(im_idx):float()
    elseif torch.isTensor(im_idx) then
      I = im_idx
    end
    I = self.image_transformer:preprocess(I)
    if flip then
      I = image.hflip(I)
    end
    local rows = I:size(2)
    local cols = I:size(3)
    feats = {}
    feats.rsp = {}
    local mtype = self.model.output:type()
    
    -- compute conv5 feature maps at different scales
    for i=1,#scales do
--      local Ir = image.scale(I,'^'..scales[i])
      local sr = rows < cols and scales[i] or math.ceil(scales[i]*rows/cols)
      local sc = rows > cols and scales[i] or math.ceil(scales[i]*cols/rows)
      local Ir = image.scale(I,sc,sr):type(mtype)
      
      local f = self.model:forward(Ir)
      
      feats.rsp[i] = torch.FloatTensor(f:size()):copy(f)
    end
    
    collectgarbage()
    collectgarbage()
    
    feats.imSize = torch.FloatTensor(I:size():totable())
    
    if self.use_cache then
      paths.mkdir(paths.dirname(cachefile))
      local f = hdf5.open(cachefile..'.h5','w')
      local options = hdf5.DataSetOptions()
      options:setChunked(128, 32, 32)
      options:setDeflate()
      feats.scales = torch.FloatTensor(scales)
      
      for i,v in pairs(feats) do
        if i == 'imSize' or i == 'scales' then
          f:write('/'..i,v)          
        elseif i == 'rsp' then
          for l,k in pairs(v) do
            f:write('/rsp/'..l,k,options)
          end
        end
      end
      
      f:close()
    end
    
    feats.scales = scales    
  end
  return feats
end

function SPP:getBestSPPScale(bbox,imSize,scales)

  local scales = scales or self.scales
  local num_scales = #scales
  if torch.isTensor(num_scales) then
    num_scales = num_scales[1]
  end
  
  local min_dim = math.min(imSize[2],imSize[3])
  
  local sz_conv_standard = self.sz_conv_standard
  local step_standard = self.step_standard

  local bestScale

  if self.train then
    -- in training, select the scales randomly
    bestScale = torch.random(1,num_scales)
  else
    local inputArea = self.inputArea
    local bboxArea = (bbox[4]-bbox[2]+1)*(bbox[3]-bbox[1]+1)
    
    local expected_scale = sz_conv_standard*step_standard*min_dim/math.sqrt(bboxArea)
    expected_scale = torch.round(expected_scale)
    
    local nbboxDiffArea = torch.Tensor(num_scales)

    for i=1,#scales do
      nbboxDiffArea[i] = math.abs(scales[i]-expected_scale)
    end
  
    bestScale = select(2,nbboxDiffArea:min(1))[1] -- index of minimum area
  
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



local function unique(bboxes)
  local idx = {}
  local is_unique = torch.ones(bboxes:size(1))
  for i=1,bboxes:size(1) do
    local b = bboxes[i]
    local n = b[1]..'_'..b[2]..'_'..b[3]..'_'..b[4]..'_'..b[5]
    if idx[n] then
      is_unique[i] = 0
    else
      idx[n] = i
    end
  end
  return is_unique
end

-- given a table with the conv5 features at different scales and bboxes in
-- the original image, project the bboxes in the conv5 space
function SPP:projectBoxes(feat, bboxes, scales)
  -- bboxes is a nx4 Tensor with candidate bounding boxes
  -- in [x1, y1, x2, y2] format
  local imSize = feat.imSize

  local scales = scales or self.scales
  local min_dim = math.min(imSize[2],imSize[3])

  local sz_conv_standard = self.sz_conv_standard
  local step_standard = self.step_standard

  local nboxes = bboxes:size(1)

  -- get best SPP scale
  local bestScale = torch.FloatTensor(nboxes)

  if self.train then
    -- in training, select the scales randomly
    bestScale:random(1,#scales)
  else
    local bboxArea = boxes.new():resize(nboxes):zero()
    bboxArea:map2(bboxes[{{},3}],bboxes[{{},1}],function(xx,xx2,xx1) return xx2-xx1+1 end)
    bboxArea:map2(bboxes[{{},4}],bboxes[{{},2}],function(xx,xx2,xx1) return xx*(xx2-xx1+1) end)

    local expected_scale = bboxArea:float():pow(-0.5):mul(sz_conv_standard*step_standard*min_dim)
    expected_scale:round()

    local nbboxDiffArea = torch.FloatTensor(#scales,nboxes)

    for i=1,#scales do
      nbboxDiffArea[i]:copy(expected_scale):add(-scales[i]):abs()
    end

    bestScale = select(2,nbboxDiffArea:min(1))[1]
  end

  local mul_factor = torch.FloatTensor(nboxes,1):copy(bestScale)
  local idx = 0
  mul_factor:apply(function(x)
                     idx = idx + 1
                     return (scales[x]-1)/(min_dim-1)
                   end)

  local bestbboxes = torch.FloatTensor(nboxes,4):copy(bboxes)
  bestbboxes:add(-1):cmul(mul_factor:expand(nboxes,4)):add(1)

  -- response boxes

  local offset0 = self.offset0
  local offset = self.offset

  local bboxes_norm = bestbboxes:clone()
  bboxes_norm[{{},{1,2}}]:add(-offset0 + offset):div(step_standard):add( 0.5)
  bboxes_norm[{{},{1,2}}]:floor():add(1)
  bboxes_norm[{{},{3,4}}]:add(-offset0 - offset):div(step_standard):add(-0.5)
  bboxes_norm[{{},{3,4}}]:ceil():add(1)

  local x0gtx1 = bboxes_norm[{{},1}]:gt(bboxes_norm[{{},3}])
  local y0gty1 = bboxes_norm[{{},2}]:gt(bboxes_norm[{{},4}])

  bboxes_norm[{{},1}][x0gtx1] = bboxes_norm[{{},1}][x0gtx1]:add(bboxes_norm[{{},3}][x0gtx1]):div(2)
  bboxes_norm[{{},3}][x0gtx1] = (bboxes_norm[{{},1}][x0gtx1])

  bboxes_norm[{{},2}][y0gty1] = bboxes_norm[{{},2}][y0gty1]:add(bboxes_norm[{{},4}][y0gty1]):div(2)
  bboxes_norm[{{},4}][y0gty1] = (bboxes_norm[{{},2}][y0gty1])

  -- remove repeated projections
  if self.dedup then
    local is_unique = unique(torch.cat(bboxes_norm,bestScale:view(-1,1),2))
    local lin = torch.range(1,is_unique:size(1)):long() -- can also use cumsum instead
    bboxes_norm = bboxes_norm:index(1,lin[is_unique])
  end
  -- clamp on boundaries

  local projected_bb = bboxes_norm:clone()

  for i=1,#scales do
    local this_scale = bestScale:eq(i)
    if this_scale:numel() > 0 then
      projected_bb[{{},2}][this_scale] = projected_bb[{{},2}][this_scale]:clamp(1,feat.rsp[i]:size(2))
      projected_bb[{{},4}][this_scale] = projected_bb[{{},4}][this_scale]:clamp(1,feat.rsp[i]:size(2))
      projected_bb[{{},1}][this_scale] = projected_bb[{{},1}][this_scale]:clamp(1,feat.rsp[i]:size(3))
      projected_bb[{{},3}][this_scale] = projected_bb[{{},3}][this_scale]:clamp(1,feat.rsp[i]:size(3))
    end
  end

  --projected_bb:floor()
  return bestScale,bestbboxes,bboxes_norm,projected_bb
end

-- don't do anything. could be the bbox regression or SVM, but I won't add it here
function SPP:postProcess(im,bbox,output)
  return output
end

function SPP:compute(model,inputs)
  local inputs_s = inputs:split(self.max_batch_size,1)

  self.output = self.output or inputs.new()

  local ttype = model.output:type()
  self.inputs = self.inputs or torch.Tensor():type(ttype)

  for idx, f in ipairs(inputs_s) do
    self.inputs:resize(f:size()):copy(f)
    local output0 = model:forward(self.inputs)
    local fs = f:size(1)
    if idx == 1 then
      local ss = output0[1]:size():totable()
      self.output:resize(inputs:size(1),table.unpack(ss))
    end
    self.output:narrow(1,(idx-1)*self.max_batch_size+1,fs):copy(output0)
  end
  return self.output
end

function SPP:type(t_type)
  self._type = t_type
  --self.spp_pooler = self.spp_pooler:type(t_type)
  return self
end

function SPP:float()
  return self:type('torch.FloatTensor')
end

function SPP:double()
  return self:type('torch.DoubleTensor')
end

function SPP:cuda()
  return self:type('torch.CudaTensor')
end
