local function createWindowBase(rec,i,j,is_bg)
  local label = is_bg == true and 0+1 or rec.label[j]+1
  local window = {i,rec.boxes[j][1],rec.boxes[j][2],
                    rec.boxes[j][3],rec.boxes[j][4],
                    label}
  return window
end

local function createWindowAngle(rec,i,j,is_bg)
  local label = is_bg == true and 0+1 or rec.label[j]+1
  --local ang = ( is_bg == false and rec.objects[rec.correspondance[j] ] ) and 
  --                  rec.objects[rec.correspondance[j] ].viewpoint.azimuth or 0
  local ang
  if is_bg == false and rec.objects[rec.correspondance[j] ] then
    if rec.objects[rec.correspondance[j] ].viewpoint.distance == '0' then
      ang = rec.objects[rec.correspondance[j] ].viewpoint.azimuth_coarse
    else
      ang = rec.objects[rec.correspondance[j] ].viewpoint.azimuth
    end
  else
    ang = 0
  end
  local window = {i,rec.boxes[j][1],rec.boxes[j][2],
                    rec.boxes[j][3],rec.boxes[j][4],
                    label,ang}
  return window
end


local argcheck = require 'argcheck'
local initcheck = argcheck{
  --pack=true,
  debug=true,
  --noordered=true,
--  {name="self",
--   type="nnf.BatchProviderBase"
--  },
--  {name="dataset",
--   type="nnf.DatasetPascal",
--   help="A dataset class" 
--  },
  {name="batch_size",
   type="number",
   help="batch size"},
  {name="fg_fraction",
   type="number",
   help="foreground fraction in batch" 
  },
  {name="fg_threshold",
   type="number",
   help="foreground threshold" 
  },
  {name="bg_threshold",
   type="number",--"table",
   help="background threshold, in the form {LO,HI}" 
  },
--  {name="createWindow",
--   type="function",
--   default=createWindowBase,
--   help="" 
--  },
  {name="do_flip",
   type="boolean",
   help="sample batches with random flips" 
  },

}


local BatchProviderBase = torch.class('nnf.BatchProviderBase')

BatchProviderBase.__init = initcheck
--[[
function BatchProviderBase:__init(dataset)

  self.dataset = dataset

  self.batch_size = 128
  self.fg_fraction = 0.25
  
  self.fg_threshold = 0.5
  self.bg_threshold = {0.0,0.5}
  
  self.createWindow = createWindowBase
  
  self.do_flip = true
  
end
--]]

function BatchProviderBase:setupData()
  local dataset = self.dataset
  local bb = {}
  local bbT = {}

  for i=0,dataset.num_classes do -- 0 because of background
    bb[i] = {}
  end

  for i=1,dataset.num_imgs do
    bbT[i] = {}
  end

  for i = 1,dataset.num_imgs do
    if dataset.num_imgs > 10 then
      xlua.progress(i,dataset.num_imgs)
    end
    
    local rec = dataset:attachProposals(i)
  
    for j=1,rec:size() do    
      local id = rec.label[j]
      local is_fg = (rec.overlap[j] >= self.fg_threshold)
      local is_bg = (not is_fg) and (rec.overlap[j] >= self.bg_threshold[1]  and
                                     rec.overlap[j] <  self.bg_threshold[2])
      if is_fg then
        local window = self.createWindow(rec,i,j,is_bg)
        table.insert(bb[1], window) -- could be id instead of 1
      elseif is_bg then
        local window = self.createWindow(rec,i,j,is_bg)
        table.insert(bb[0], window)
      end
      
    end
    
    for j=0,dataset.num_classes do -- 0 because of background
      if #bb[j] > 0 then
        bbT[i][j] = torch.FloatTensor(bb[j])
      end
    end
        
    bb = {}
    for i=0,dataset.num_classes do -- 0 because of background
      bb[i] = {}
    end
    collectgarbage()
  end
  self.bboxes = bbT
  --return bbT
end

function BatchProviderBase:getBatch(input,target)
  error("You can't use BatchProviderBase")
  return input,target
end

