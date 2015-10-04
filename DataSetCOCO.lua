--local json = require 'dkjson'

local DataSetCOCO,parent = torch.class('nnf.DataSetCOCO', 'nnf.DataSetDetection')

function DataSetCOCO:__init(annFile)
  self.image_set = nil
  self.dataset_name = 'COCO'

  local timer = torch.Timer()
  local localtimer = torch.Timer()
  print('Preparing COCO dataset...')
  --[[
  if type(annFile) == 'string' then
    local f = io.open(annFile)
    local str = f:read('*all')
    f:close()

    self.data = json.decode(str)

  else
    self.data = torch.load(annFile)
  end
  --]]
  self.data = torch.load('coco_val.t7')
  print(('  Loaded annotations file in %.2fs'):format(localtimer:time().real))
  localtimer:reset()

  -- mapping images
  local img_idx = {}
  local img_idx_map = {}
  for i = 1, #self.data.images do
    table.insert(img_idx,self.data.images[i].id)
    img_idx_map[self.data.images[i].id] = i
  end
  print(('  Mapped images in %.4fs'):format(localtimer:time().real))
  localtimer:reset()

  -- mapping annotations
  local ann = self.data.annotations
  local o = {}

  for k, v in ipairs(ann) do
    table.insert(o,v.image_id*1e10 + v.category_id)
  end
  o = torch.LongTensor(o)
  local _,ox = o:sort()
  local o_data = ox:data()
  local temp_ann = {}
  for i=1 , o:size(1) do
    table.insert(temp_ann, ann[ox[i] ])
  end
  self.data.annotations = temp_ann
  
  local ann_idx = {}
  local ann_idx_map = {}
  local ann_img_idx = {}
  local img_ann_idx_map = {}
  for k,v in ipairs(temp_ann) do
    table.insert(ann_idx, v.id)
    ann_idx_map[v.id] = k
    table.insert(ann_img_idx, v.image_id)
    if not img_ann_idx_map[v.image_id] then
      img_ann_idx_map[v.image_id] = {}
    end
    table.insert(img_ann_idx_map[v.image_id],v.id)
  end

  self.inds = {img_idx = img_idx,
               img_idx_map = img_idx_map,
               ann_idx = ann_idx,
               ann_idx_map = ann_idx_map,
               ann_img_idx = ann_img_idx,
               img_ann_idx_map = img_ann_idx_map
             }
  print(('  Mapped annotations in %.4fs'):format(localtimer:time().real))
  localtimer:reset()

  -- mapping classes
  self.classes = {}
  self.class_to_id = {}
  self.class_cont = {}
  self.class_cont_map = {}
  self.num_classes = 0
  for k,v in ipairs(self.data.categories) do
    self.classes[v.id] = v.name
    self.class_to_id[v.name] = v.id
    table.insert(self.class_cont,v.id)
    self.class_cont_map[v.id] = k
    self.num_classes = self.num_classes + 1
  end

  print(('  Total elapsed time: %.4fs'):format(timer:time().real))

end

function DataSetCOCO:getImage(i)
  local file_name = self.images[i].file_name
  return image.load(paths.concat(self.imgpath,file_name),3,'float')
end

function DataSetCOCO:getAnnotation(i)
  local ann = {object = {}}
  local im_id = self.inds.img_idx[i]
  local ann_id = self.inds.img_ann_idx_map[im_id] or {}
  for k,v in ipairs(ann_id) do
    local lann = self.data.annotations[self.inds.ann_idx_map[v] ]
    local bbox = {xmin=lann.bbox[1]+1,ymin=lann.bbox[2]+1,
                  xmax=lann.bbox[1]+lann.bbox[3]+1,
                  ymax=lann.bbox[2]+lann.bbox[4]+1,
                 }
    local obj = {bndbox=bbox,
                 class=lann.category_id,
                 difficult = '0',
                 name = self.classes[lann.category_id]
                }
    table.insert(ann.object,obj)
  end
  return ann
end

function DataSetCOCO:getGTBoxes(i)
  local anno = self:getAnnotation(i)
  local valid_objects = {}
  local gt_boxes = torch.IntTensor()
  local gt_classes = {}

  if self.with_hard_samples then -- inversed with respect to RCNN code
    for idx,obj in ipairs(anno.object) do
      if self.class_to_id[obj.name] then -- to allow a subset of the classes
        table.insert(valid_objects,idx)
      end
    end
  else
    for idx,obj in ipairs(anno.object) do
      if obj.difficult == '0' and self.class_to_id[obj.name] then
        table.insert(valid_objects,idx)
      end
    end
  end
  
  gt_boxes:resize(#valid_objects,4)
  for idx0,idx in ipairs(valid_objects) do
    gt_boxes[idx0][1] = anno.object[idx].bndbox.xmin
    gt_boxes[idx0][2] = anno.object[idx].bndbox.ymin
    gt_boxes[idx0][3] = anno.object[idx].bndbox.xmax
    gt_boxes[idx0][4] = anno.object[idx].bndbox.ymax
    
    table.insert(gt_classes,self.class_cont_map[anno.object[idx].class])
  end

  return gt_boxes,gt_classes,valid_objects,anno
 
end


