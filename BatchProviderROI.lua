local BatchProviderROI, parent = torch.class('nnf.BatchProviderROI','nnf.BatchProvider')

function BatchProviderROI:__init(dataset)
  local fp = {dataset=dataset}
  parent:__init(fp)
  self.imgs_per_batch = 2
  self.scale = 600
  self.max_size = 1000
  self.image_transformer = nnf.ImageTransformer{}
end

-- setup is the same

function BatchProviderROI:permuteIdx()
  local fg_num_each  = self.fg_num_each
  local bg_num_each  = self.bg_num_each
  local fg_num_total = self.fg_num_total
  local bg_num_total = self.bg_num_total
  local total_img    = self.dataset:size()
  local imgs_per_batch = self.imgs_per_batch

  self._cur = self._cur or math.huge

  if self._cur + imgs_per_batch > total_img  then
    self._perm = torch.randperm(total_img)
    self._cur = 1
  end

  local img_idx      = self._perm[{{self._cur,self._cur + self.imgs_per_batch - 1}}]
  self._cur = self._cur + self.imgs_per_batch

  local img_idx_end  = imgs_per_batch
 --[[ 
  local fg_windows = {}
  local bg_windows = {}
  for i=1,img_idx_end do
    local curr_idx = img_idx[i]
    bg_windows[i] = {}
    if self.bboxes[curr_idx][0] then
      for j=1,self.bboxes[curr_idx][0]:size(1) do
        table.insert(bg_windows[i],{curr_idx,j})
      end
    end
    fg_windows[i] = {}
    if self.bboxes[curr_idx][1] then
      for j=1,self.bboxes[curr_idx][1]:size(1) do
        table.insert(fg_windows[i],{curr_idx,j})
      end
    end
  end
  --]]
  local opts = {img_idx=img_idx,img_idx_end=img_idx_end}
  return fg_windows,bg_windows,opts

end

function BatchProviderROI:selectBBoxes(fg_windows,bg_windows)
  local fg_w = {}
  local bg_w = {}

  for im=1,self.imgs_per_batch do

    fg_w[im] = {}
    bg_w[im] = {}

    local window_idx = torch.randperm(#bg_windows[im])
    for i=1,math.min(self.bg_num_each,#bg_windows[im]) do
      local curr_idx = bg_windows[im][window_idx[i] ][1]
      local position = bg_windows[im][window_idx[i] ][2]
      local dd = self.bboxes[curr_idx][0][position]
      table.insert(bg_w[im],dd)
    end

    window_idx = torch.randperm(#fg_windows[im])
    for i=1,math.min(self.fg_num_each,#fg_windows[im]) do
      local curr_idx = fg_windows[im][window_idx[i] ][1]
      local position = fg_windows[im][window_idx[i] ][2]
      local dd = self.bboxes[curr_idx][1][position]
      table.insert(fg_w[im],dd)
    end
  
  end

  return fg_w,bg_w
end

local function getImages(self,img_ids,images)
  local dataset = self.dataset
  local num_images = img_ids:size(1)

  local imgs = {}
  local im_sizes = {}
  local im_scales = {}

  for i=1,num_images do
    local im = dataset:getImage(img_ids[i])
    im = self.image_transformer:preprocess(im)
    local im_size = im[1]:size()
    local im_size_min = math.min(im_size[1],im_size[2])
    local im_size_max = math.max(im_size[1],im_size[2])
    local im_scale = self.scale/im_size_min
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
  images:resize(num_images,3,max_shape[1],max_shape[2])
  for i=1,num_images do
    images[i][{{},{1,imgs[i]:size(2)},{1,imgs[i]:size(3)}}]:copy(imgs[i])
  end
  return im_scales
end


function BatchProviderROI:getBatch(batches,targets)
  local dataset = self.dataset
  
  self.fg_num_each = self.fg_fraction * self.batch_size
  self.bg_num_each = self.batch_size - self.fg_num_each
  --self.fg_num_total = self.fg_num_each * self.iter_per_batch
  --self.bg_num_total = self.bg_num_each * self.iter_per_batch
  
  local fg_windows,bg_windows,opts = self:permuteIdx()
  --local fg_w,bg_w = self:selectBBoxes(fg_windows,bg_windows)
  
  local batches = batches or {torch.FloatTensor(),torch.FloatTensor()}
  local targets = targets or torch.IntTensor()
  
 -- batches[1]:resize(self.batch_size,unpack(self.batch_dim))
  local im_scales = getImages(self,opts.img_idx,batches[1])
  batches[2]:resize(self.batch_size,unpack(self.batch_dim))
  targets:resize(self.batch_size,self.target_dim)
  
  return batches, targets
end
