local RCNN = torch.class('nnf.RCNN')

function RCNN:__init(dataset)
  self.dataset = dataset
  
  self.crop_size = 227
  self.image_mean = nil
  self.padding = 16
  self.use_square = false
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

local function round(num)
  if num >= 0 then 
    return math.floor(num+.5)
  else 
    return math.ceil(num-.5)
  end
end

function RCNN:getCrop(im_idx,bbox,flip)
  -- suppose I is in BGR, as image_mean
  -- [x1 y1 x2 y2] order
  local flip = flip==nil and false or flip
  
  if self.curr_im_idx ~= im_idx or self.curr_doflip ~= flip then
    self.curr_im_idx = im_idx
    self.curr_im_feats = self.dataset:getImage(im_idx):float()
    self.curr_im_feats = prepareImage(self.curr_im_feats)
    if flip then
      self.curr_im_feats = image.hflip(self.curr_im_feats)
    end
    self.curr_doflip = flip
  end
  
  local I = self.curr_im_feats
  local bbox = bbox
  
  if flip then
    local tt = bbox[1]
    bbox[1] = I:size(3)-bbox[3]+1
    bbox[3] = I:size(3)-tt     +1
  end
  
  local crop_size = self.crop_size
  local image_mean = self.image_mean
  local padding = self.padding
  local use_square = self.use_square

  local pad_w = 0;
  local pad_h = 0;
  local crop_width = crop_size;
  local crop_height = crop_size;

  --local bbox = {bbox[2],bbox[1],bbox[4],bbox[3]}

  ------
  if padding > 0 or use_square then
    local scale = crop_size/(crop_size - padding*2)
    local half_height = (bbox[4]-bbox[2]+1)/2
    local half_width = (bbox[3]-bbox[1]+1)/2
    local center = {bbox[1]+half_width, bbox[2]+half_height}
    if use_square then
      -- make the box a tight square
      if half_height > half_width then
        half_width = half_height;
      else
        half_height = half_width;
      end
    end
    bbox[1] = round(center[1] - half_width  * scale)
    bbox[2] = round(center[2] - half_height * scale)
    bbox[3] = round(center[1] + half_width  * scale)
    bbox[4] = round(center[2] + half_height * scale)

    local unclipped_height = bbox[4]-bbox[2]+1;
    local unclipped_width = bbox[3]-bbox[1]+1;
    
    local pad_x1 = math.max(0, 1 - bbox[1]);
    local pad_y1 = math.max(0, 1 - bbox[2]);
    -- clipped bbox
    bbox[1] = math.max(1, bbox[1]);
    bbox[2] = math.max(1, bbox[2]);
    bbox[3] = math.min(I:size(3), bbox[3]);
    bbox[4] = math.min(I:size(2), bbox[4]);
    local clipped_height = bbox[4]-bbox[2]+1;
    local clipped_width = bbox[3]-bbox[1]+1;
    local scale_x = crop_size/unclipped_width;
    local scale_y = crop_size/unclipped_height;
    crop_width = round(clipped_width*scale_x);
    crop_height = round(clipped_height*scale_y);
    pad_x1 = round(pad_x1*scale_x);
    pad_y1 = round(pad_y1*scale_y);

    pad_h = pad_y1;
    pad_w = pad_x1;

    if pad_y1 + crop_height > crop_size then
      crop_height = crop_size - pad_y1;
    end
    if pad_x1 + crop_width > crop_size then
      crop_width = crop_size - pad_x1;
    end
  end -- padding > 0 || square
  ------

  --local patch = image.crop(I,bbox[1],bbox[2],bbox[3],bbox[4]);
  local patch = image.crop(I,bbox[1],bbox[2],bbox[3],bbox[4]):float();
  local tmp = image.scale(patch,crop_width,crop_height,'bilinear');

  if image_mean then
    tmp = tmp - image_mean[{{},{pad_h+1,pad_h+crop_height},
                               {pad_w+1,pad_w+crop_width}}]
  end

  --patch = torch.zeros(3,crop_size,crop_size):typeAs(I)
  patch = torch.zeros(3,crop_size,crop_size):float()

  patch[{{},{pad_h+1,pad_h+crop_height}, {pad_w+1,pad_w+crop_width}}] = tmp

  return patch

end

function RCNN:getFeature(im_idx,bbox,flip)
  local flip = flip==nil and false or flip
  
  local crop_feat = self:getCrop(im_idx,bbox,flip)
  
  return crop_feat
end


