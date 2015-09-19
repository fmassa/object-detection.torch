local ImageDetect = torch.class('nnf.ImageDetect')
local recursiveResizeAsCopyTyped = utils.recursiveResizeAsCopyTyped

function ImageDetect:__init(model, feat_provider)
  self.model = model
  self.feat_provider = feat_provider
  --self.sm = nn.SoftMax():cuda()
end

-- supposes boxes is in [x1,y1,x2,y2] format
function ImageDetect:detect(im,boxes)
  local ttype = self.model.output:type()

  local inputs = self.feat_provider:getFeature(im,boxes)
  self.inputs,inputs = recursiveResizeAsCopyTyped(self.inputs,inputs,ttype)

  local output0 = self.model:forward(self.inputs)
  local output = self.feat_provider:postProcess(im,boxes,output0)
  --self.sm:forward(output0)

  self.output,output = recursiveResizeAsCopyTyped(self.output,output,'torch.FloatTensor')
  return self.output
end
