local matio = require 'matio'
local argcheck = dofile'argcheck.lua'--require 'argcheck'
local xml = require 'xml'
local utilities = paths.dofile('utils.lua')
local concat = utilities.concat
local boxoverlap = utilities.boxoverlap

matio.use_lua_strings = true

local DataSetPascal,parent = torch.class('nnf.DataSetPascal', 'nnf.DataSetDetection')

local function lines_from(file)
-- get all lines from a file, returns an empty 
-- list/table if the file does not exist
  if not paths.filep(file) then return {} end
  local lines = {}
  for line in io.lines(file) do 
    table.insert(lines,line)
  end
  return lines
end

--


local env = require 'argcheck.env' -- retrieve argcheck environement
-- this is the default type function
-- which can be overrided by the user
function env.istype(obj, typename)
  local t = torch.type(obj)
  if t:find('torch.*Tensor') then
    return 'torch.Tensor' == typename
  end
  return torch.type(obj) == typename
end


local initcheck = argcheck{
  pack=true,
  noordered=true,
  help=[[
    A dataset class for object detection in Pascal-like datasets.
]],
  {name="image_set",
   type="string",
   help="ImageSet name"},
  {name="datadir",
   type="string",
   help="Path to dataset (follows standard Pascal folder structure)",
   check=function(datadir)
           return paths.dirp(datadir)
         end},
  {name="classes",
   type="table",
   help="Classes to be considered",
   default = {'aeroplane','bicycle','bird','boat','bottle','bus','car',
              'cat','chair','cow','diningtable','dog','horse','motorbike',
              'person','pottedplant','sheep','sofa','train','tvmonitor'},
   check=function(classes)
      local out = true;
      for k,v in ipairs(classes) do
        if type(v) ~= 'string' then
          print('classes can only be of string input');
          out = false
          break
        end
      end
      return out
     end},
  {name="imgsetpath",
   type="string",
   help="Path to the ImageSet file",
   opt = true},
  {name="with_hard_samples",
   type="boolean",
   help="Use difficult samples in the proposals",
   opt = true},
  {name="year",
   type="number",
   help="Year of the dataset (for Pascal)",
   opt = true},
  {name="roidbdir",
   type="string",
   help="Path to the folder with the bounding boxes",
   opt = true},
  {name="annopath",
   type="string",
   help="Path to the annotations",
   opt = true},
  {name="imgpath",
   type="string",
   help="Path to the images",
   opt = true},
  {name="roidbfile",
   type="string",
   help="Mat file with the bounding boxes",
   opt = true},
  {name="dataset_name",
   type="string",
   help="Name of the dataset",
   opt = true}--[[,
  {name="image",
   type="torch.Tensor",
   help="Dataset of one single image",
   opt = true}]]
}

function DataSetPascal:__init(...)
  parent.__init(self)
  local args = initcheck(...)
  print(args)
  for k,v in pairs(args) do self[k] = v end
  
  local image_set = self.image_set

  if not self.year then
    self.year = 2007
  end
  local year = self.year
    
  if not self.dataset_name then
    self.dataset_name = 'VOC'..year
  end
  
  if not self.annopath then
    self.annopath = paths.concat(self.datadir,self.dataset_name,'Annotations','%s.xml')
  end
  if not self.imgpath then
    self.imgpath = paths.concat(self.datadir,self.dataset_name,'JPEGImages','%s.jpg')
  end
  if not self.imgsetpath then
    self.imgsetpath = paths.concat(self.datadir,self.dataset_name,'ImageSets','Main','%s.txt')
  end
    
  if not self.roidbfile and self.roidbdir then
    self.roidbfile = paths.concat(self.roidbdir,'voc_'..year..'_'..image_set..'.mat')
  end
  
  self.num_classes = #self.classes
  self.class_to_id = {}
  for i,v in ipairs(self.classes) do
    self.class_to_id[v] = i
  end
    
  self.img_ids = lines_from(string.format(self.imgsetpath,image_set))
  self.num_imgs = #self.img_ids
  
  -- 
  if self.image then
    self.img_ids = {}
  end
  
  --[[
  self.sizes = {}
  print('Getting Image Sizes')
  for i=1,#self.img_ids do
    xlua.progress(i,#self.img_ids)
    local imp = string.format(self.imgpath,self.img_ids[i])
    table.insert(self.sizes,{image.getJPGsize(imp)})
    if i%100 == 0 then
      collectgarbage()
    end
  end
  self.sizes = torch.IntTensor(self.sizes)
  ]]
  
end

function DataSetPascal:size()
  return #self.img_ids
end

function DataSetPascal:getImage(i)
  return image.load(string.format(self.imgpath,self.img_ids[i]),3,'float')
end


local function parsePascalAnnotation(ann,ind,parent)
  local res = {}
  for i,j in ipairs(ann) do
    if #j == 1 then
      res[j.xml] = j[1]
    else
      local sub = parsePascalAnnotation(j,i,j.xml)
      if not res[j.xml] then
        res[j.xml] = sub
      elseif #res[j.xml] == 0 then
        res[j.xml] = {res[j.xml]}
        table.insert(res[j.xml],sub)
      else
        table.insert(res[j.xml],sub)
      end
    end
  end
  return res
end

function DataSetPascal:getAnnotation(i)
  local ann = xml.loadpath(string.format(self.annopath,self.img_ids[i]))
  local parsed = parsePascalAnnotation(ann,1,{})
  if parsed.object and #parsed.object == 0 then
    parsed.object = {parsed.object}
  end
  return parsed
end

function DataSetPascal:__tostring__()
  local str = torch.type(self)
  str = str .. '\n  Dataset Name: ' .. self.dataset_name
  str = str .. '\n  ImageSet: '.. self.image_set
  str = str .. '\n  Number of images: '.. self:size()
  str = str .. '\n  Classes:'
  for k,v in ipairs(self.classes) do
    str = str .. '\n    '..v
  end
  return str
end


function DataSetPascal:loadROIDB()
  if self.roidb then
    return
  end
  local roidbfile = self.roidbfile
  
  assert(roidbfile and paths.filep(roidbfile),'Need to specify the bounding boxes file')
  
  local dt = matio.load(roidbfile)
  
  self.roidb = {}
  local img2roidb = {}
  -- compat: change coordinate order from [y1 x1 y2 x2] to [x1 y1 x2 y2]
  for i=1,#dt.images do
    img2roidb[dt.images[i] ] = i
  end
  
  for i=1,self:size() do
    if dt.boxes[img2roidb[self.img_ids[i] ] ]:size(2) ~= 4 then
      table.insert(self.roidb,torch.IntTensor(0,4))
    else
      table.insert(self.roidb, dt.boxes[img2roidb[self.img_ids[i] ] ]:index(2,torch.LongTensor{2,1,4,3}):int())
    end
  end
  
end

function DataSetPascal:getROIBoxes(i)
  if not self.roidb then
    self:loadROIDB()
  end
  return self.roidb[i]--self.roidb[self.img2roidb[self.img_ids[i] ] ]
end

function DataSetPascal:getGTBoxes(i)
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
    
    table.insert(gt_classes,self.class_to_id[anno.object[idx].name])
  end

  return gt_boxes,gt_classes,valid_objects,anno
 
end

function DataSetPascal:createROIs()
  if self.rois then
    return
  end
  self.rois = {}
  for i=1,self.num_imgs do
    table.insert(self.rois,self:attachProposals(i))
    if i%500 == 0 then
      xlua.progress(i,self.num_imgs)
      collectgarbage()
    end
  end
  xlua.progress(self.num_imgs,self.num_imgs)
end
