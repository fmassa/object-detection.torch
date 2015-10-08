local argcheck = require 'argcheck'
local flipBoundingBoxes = paths.dofile('utils.lua').flipBoundingBoxes

local argcheck = require 'argcheck'
local initcheck = argcheck{
  pack=true,
  noordered=true,
  {name="crop_size",
   type="number",
   default=227,
   help="crop size"},
  {name="padding",
   type="number",
   default=16,
   help="context padding"},
  {name="use_square",
   type="boolean",
   default=false,
   help="force square crops"},
  {name="image_transformer",
   type="nnf.ImageTransformer",
   default=nnf.ImageTransformer{},
   help="Class to preprocess input images"},
  {name="max_batch_size",
   type="number",
   default=128,
   help="maximum size of batches during evaluation"},
  {name="dataset",
   type="nnf.DataSetPascal", -- change to allow other datasets
   opt=true,
   help="A dataset class"},
}


local RCNN = torch.class('nnf.RCNN')
RCNN._isFeatureProvider = true

function RCNN:__init(...)
--  self.image_transformer = nnf.ImageTransformer{
--                                  mean_pix={123.68/255,116.779/255,103.939/255}}
  
  self.image_mean = nil

  local opts = initcheck(...)
  for k,v in pairs(opts) do self[k] = v end

  self.output_size = {3,self.crop_size,self.crop_size}
  self.train = true
end

function RCNN:training()
  self.train = true
end

function RCNN:evaluate()
  self.train = false
end

function RCNN:getCrop(output,I,bbox)
  -- suppose I is in BGR, as image_mean
  -- [x1 y1 x2 y2] order

  local crop_size = self.crop_size
  local image_mean = self.image_mean
  local padding = self.padding
  local use_square = self.use_square

  local pad_w = 0;
  local pad_h = 0;
  local crop_width = crop_size;
  local crop_height = crop_size;

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
    bbox[1] = torch.round(center[1] - half_width  * scale)
    bbox[2] = torch.round(center[2] - half_height * scale)
    bbox[3] = torch.round(center[1] + half_width  * scale)
    bbox[4] = torch.round(center[2] + half_height * scale)

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
    crop_width = torch.round(clipped_width*scale_x);
    crop_height = torch.round(clipped_height*scale_y);
    pad_x1 = torch.round(pad_x1*scale_x);
    pad_y1 = torch.round(pad_y1*scale_y);

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

  local patch = I[{{},{bbox[2],bbox[4]},{bbox[1],bbox[3]}}]
  self._crop = self._crop or torch.FloatTensor(3,self.crop_size,self.crop_size)
  self._crop:resize(3,crop_height,crop_width)
  image.scale(self._crop,patch,'bilinear');
  local tmp = self._crop

  if image_mean then
    tmp:add(-1,image_mean[{{},{pad_h+1,pad_h+crop_height},
                              {pad_w+1,pad_w+crop_width}}])
  end

  output[{{},{pad_h+1,pad_h+crop_height}, {pad_w+1,pad_w+crop_width}}] = tmp

  return output

end

function RCNN:getFeature(im_idx,bbox,flip)
  local flip = flip==nil and false or flip
  
  local crop_feat = self:getCrop(im_idx,bbox,flip)
  
  return crop_feat
end

function RCNN:getFeature(im,bbox,flip)
  local flip = flip==nil and false or flip

  if type(im) == 'number' then
    assert(self.dataset, 'you must provide a dataset if using numeric indices')
    im = self.dataset:getImage(im)
  end
  if type(bbox) == 'table' then
    bbox = torch.FloatTensor(bbox)
  end
  
  im = self.image_transformer:preprocess(im)
  bbox = bbox:dim() == 1 and bbox:view(1,-1) or bbox
  local num_boxes = bbox:size(1)

  if flip then
    im = image.hflip(im)
    flipBoundingBoxes(bbox,im:size(3))
  end

  self._feat = self._feat or torch.FloatTensor()

  self._feat:resize(num_boxes,table.unpack(self.output_size)):zero()

  -- use threads to make it faster
  for i=1,num_boxes do
    self:getCrop(self._feat[i],im,bbox[i])
  end
  
  return self._feat
end

-- don't do anything. could be the bbox regression or SVM, but I won't add it here
function RCNN:postProcess(im,bbox,output)
  return output,bbox
end

function RCNN:compute(model,inputs)
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
