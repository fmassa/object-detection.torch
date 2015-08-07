local ROIDataLayer,parent = torch.class('nnf.ROIDataLayer','nnf.BatchProvider')

function ROIDataLayer:__init(dataset)
  parent.__init(self)
  self.dataset = dataset
  self.image_transformer
  self.imgs_per_batch = 2
  self.scale = 600
  self.max_size = 1000
end

local function shuffle_roidb_inds(self)
  self._perm = torch.randperm(self.dataset:size())
  self._curr = 0
end

local function get_next_minibatch_inds(self)
  if self._cur + cfg.TRAIN.IMS_PER_BATCH >= len(self._roidb) then
    self:shuffle_roidb_inds()
  end

  local db_inds = self._perm[{{self._cur,self._cur + self.imgs_per_batch}}]
  self._cur = self._cur + self.imgs_per_batch
  return db_inds
end


function ROIDataLayer:getBatch()
  local dataset = self.dataset
  local img_ids = self:get_next_minibatch_inds()
  
  local num_images = img_ids:size(1)
  local imgs = {}
  local im_sizes = {}
  -- get images
  -- prep_im_for_blob
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
  end
  -- create single tensor with all images, padding with zero for different sizes
  im_sizes = torch.IntTensor(im_sizes)
  local max_shape = im_sizes:max(1)
  local images = torch.FloatTensor(num_images,3,max_shape[1],max_shape[2])
  for i=1,num_images do
    images[i][{{1,imgs[i]:size(2)},{imgs[i]:size(3)}}]:copy(imgs[i])
  end

  return images
end


