local hdf5 = require 'hdf5'

local SPP = torch.class('nnf.SPP')

--TODO vectorize code ?
function SPP:__init(dataset,model)

  self.dataset = dataset
  self.model = model
  self.spp_pooler = inn.SpatialPyramidPooling({{1,1},{2,2},{3,3},{6,6}}):float()
  self.image_transformer = nnf.ImageTransformer{}

-- paper=864, their code=874 
  self.scales = {480,576,688,874,1200} -- 874
  self.randomscale = true
  
  self.sz_conv_standard = 13
  self.step_standard = 16
  self.offset0 = 21
  self.offset = 6.5
  
  self.inputArea = 224^2
  
  self.use_cache = true

  self.cachedir = nil
  
end


function SPP:getCrop(im_idx,bbox,flip)
  local flip = flip or false
  
  if self.curr_im_idx ~= im_idx or self.curr_doflip ~= flip then
    self.curr_im_idx = im_idx
    self.curr_im_feats = self:getConv5(im_idx,flip)
    self.curr_doflip = flip
  end
  
  local bbox = bbox
  if flip then
    local tt = bbox[1]
    bbox[1] = self.curr_im_feats.imSize[3]-bbox[3]+1
    bbox[3] = self.curr_im_feats.imSize[3]-tt     +1
  end
  
  local bestScale,bestBbox = self:getBestSPPScale(bbox,self.curr_im_feats.imSize,self.curr_im_feats.scales)
  local box_norm = self:getResposeBoxes(bestBbox)

  local crop_feat = self:getCroppedFeat(self.curr_im_feats.rsp[bestScale],box_norm)
  
  return crop_feat  
end

function SPP:getFeature(im_idx,bbox,flip)
  local flip = flip or false

  local crop_feat = self:getCrop(im_idx,bbox,flip)

  local feat = self.spp_pooler:forward(crop_feat)

  return feat
end

function SPP:getConv5(im_idx,flip)
  local scales = self.scales
  local flip = flip or false
  local cachedir = self.cachedir
  
  assert(cachedir or (not self.use_cache), 
         'Need to set a folder to save the conv5 features')
  
  if not cachedir then
    cachedir = ''
  end
  
  local cachefile = paths.concat(self.cachedir,self.dataset.img_ids[im_idx])

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
    local I = self.dataset:getImage(im_idx):float()
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

  if self.randomscale then
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
