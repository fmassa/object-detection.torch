local ROIPooling,parent = torch.class('nnf.ROIPooling','nn.Module')

function ROIPooling:__init(W,H)
  parent.__init(self)
  self.W = W
  self.H = H
  self.pooler = {}--nn.SpatialAdaptiveMaxPooling(W,H)
end

-- not for batches for the moment
function ROIPooling:updateOutput(input)
  local data = input[1]
  local rois = input[2]
  local num_rois = rois:size(1)
  local s = data:size()
  local ss = s:size(1)
  self.output:resize(num_rois,s[ss-2],self.H,self.W)

  if #self.pooler < num_rois then
    local diff = num_rois - #self.pooler
    for i=1,diff do
      table.insert(self.pooler,nn.SpatialAdaptiveMaxPooling(self.W,self.H))
    end
  end

  for i=1,num_rois do
    local roi = rois[i]
    local im = data[{{},{roi[2],roi[4]},{roi[1],roi[3]}}]
    self.output[i] = self.pooler[i]:forward(im)
  end
  return self.output
end

function ROIPooling:updateGradInput(input,gradOutput)
  local data = input[1]
  local rois = input[2]
  local num_rois = rois:size(1)
  local s = data:size()
  local ss = s:size(1)
  self.gradInput:resizeAs(data):zero()

  for i=1,num_rois do
    local roi = rois[i]
    local r = {{},{roi[2],roi[3]},{roi[1],roi[3]}}
    local im = data[r]
    local g  = self.pooler[i]:backward(im,gradOutput[i])
    self.gradInput[r]:add(g)
  end
  return self.gradInput

end
