local ImageDetect = torch.class('nnf.ImageDetect')

function ImageDetect:__init(model)
  self.model = model
  self.image_transformer = nnf.ImageTransformer{mean_pix={102.9801,115.9465,122.7717},
                                              raw_scale = 255,
                                              swap = {3,2,1}}
  self.scale = {600}
  self.max_size = 1000
  self.sm = nn.SoftMax():cuda()
end


local function getImages(self,images,im)
  local num_scales = #self.scale

  local imgs = {}
  local im_sizes = {}
  local im_scales = {}

  im = self.image_transformer:preprocess(im)

  local im_size = im[1]:size()
  local im_size_min = math.min(im_size[1],im_size[2])
  local im_size_max = math.max(im_size[1],im_size[2])
  for i=1,num_scales do
    local im_scale = self.scale[i]/im_size_min
    if torch.round(im_scale*im_size_max) > self.max_size then
      im_scale = self.max_size/im_size_max
    end
    local im_s = {im_size[1]*im_scale,im_size[2]*im_scale}
    table.insert(imgs,image.scale(im,im_s[2],im_s[1]))
    table.insert(im_sizes,im_s)
    table.insert(im_scales,im_scale)
  end
  -- create single tensor with all images, padding with zero for different sizes
  im_sizes = torch.IntTensor(im_sizes)
  local max_shape = im_sizes:max(1)[1]
  images:resize(num_scales,3,max_shape[1],max_shape[2]):zero()
  for i=1,num_scales do
    images[i][{{},{1,imgs[i]:size(2)},{1,imgs[i]:size(3)}}]:copy(imgs[i])
  end
  return im_scales
end

local function project_im_rois(im_rois,scales)
  local levels
  local rois = torch.FloatTensor()
  if #scales > 1 then
    local scales = torch.FloatTensor(scales)
    local widths = im_rois[{{},3}] - im_rois[{{},1}] + 1
    local heights = im_rois[{{},4}] - im_rois[{{}, 2}] + 1

    local areas = widths * heights
    local scaled_areas = areas:view(-1,1) * torch.pow(scales:view(1,-1),2)
    local diff_areas = torch.abs(scaled_areas - 224 * 224)
    levels = select(2, diff_areas:min(2))
  else
    levels = torch.FloatTensor()
    rois:resize(im_rois:size(1),5)
    rois[{{},1}]:fill(1)
    rois[{{},{2,5}}]:copy(im_rois):add(-1):mul(scales[1]):add(1)
  end

  return rois

end

-- supposes boxes is in [x1,y1,x2,y2] format
function ImageDetect:detect(im,boxes)
  local inputs = {torch.FloatTensor(),torch.FloatTensor()}
  local im_scales = getImages(self,inputs[1],im)
  inputs[2] = project_im_rois(boxes,im_scales)

  local inputs_cuda =  {torch.CudaTensor(),torch.CudaTensor()}
  inputs_cuda[1]:resize(inputs[1]:size()):copy(inputs[1])
  inputs_cuda[2]:resize(inputs[2]:size()):copy(inputs[2])
  local output0 = self.model:forward(inputs_cuda)
  local output = self.sm:forward(output0):float()
  --[[
  for i=1,#im_scales do
    local dd = boxes:clone()
    dd:add(-1):mul(im_scale[i]):add(1)

  end
  --]]
  return output
end
