local ImageDetect = torch.class('nnf.ImageDetect')
local recursiveResizeAsCopyTyped = paths.dofile('utils.lua').recursiveResizeAsCopyTyped

function ImageDetect:__init(model, feat_provider)
  self.model = model
  self.feat_provider = feat_provider
  --self.sm = nn.SoftMax():cuda()
end

-- supposes boxes is in [x1,y1,x2,y2] format
function ImageDetect:detect(im,boxes)
  local feat_provider = self.feat_provider
  local ttype = self.model.output:type()

  local inputs = feat_provider:getFeature(im,boxes)
  self.inputs,inputs = recursiveResizeAsCopyTyped(self.inputs,inputs,ttype)

  local output0 = feat_provider:compute(self.model, self.inputs)
  local output = feat_provider:postProcess(im,boxes,output0)
  --self.sm:forward(output0)

  self.output,output = recursiveResizeAsCopyTyped(self.output,output,'torch.FloatTensor')
  return self.output
end
