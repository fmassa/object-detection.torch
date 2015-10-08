dofile 'test_utils.lua'

detect1 = nnf.ImageDetect(model1,fp1)
detect = nnf.ImageDetect(model,fp2)


--------------------------------------------------------------------------------
-- define batch providers
--------------------------------------------------------------------------------

bp1 = nnf.BatchProvider{dataset=ds,feat_provider=fp1}
bp1.nTimesMoreData = 2
bp1.iter_per_batch = 10
bp2 = nnf.BatchProviderROI{dataset=ds,feat_provider=fp2}

bp1.bboxes = torch.load('tests/bproibox.t7')
bp2.bboxes = torch.load('tests/bproibox.t7')

print('test1')
b,t = bp1:getBatch()
print('test2')
b,t = bp2:getBatch()

-- mixing does not work for the moment, as FRCNN accepts a set of images as input
-- whereas RCNN and SPP supposes that only one image is provided at a time
--[[
bp3 = nnf.BatchProviderROI(ds)
bp3.bboxes = torch.load('tests/bproibox.t7')
bp3.feat_provider = fp1
print('test3')
b,t = bp3:getBatch()
--]]
--------------------------------------------------------------------------------
--
--------------------------------------------------------------------------------

idx = 100
im = ds:getImage(idx)
boxes = ds:getROIBoxes(idx)

--output = detect1:detect(im,boxes)
--output0 = detect:detect(im,boxes)

--------------------------------------------------------------------------------
-- compare old and new SPP implementations for the cropping
--------------------------------------------------------------------------------
--[[
output_old = {}
for i=1,boxes:size(1) do
  tt0 = fp3:getCrop_old(im,boxes[i])
  output_old[i] = tt0
end

output_new = fp3:getCrop(im,boxes) --[881]

for i=1,boxes:size(1) do
  assert(output_old[i]:eq(output_new[i]):all(),'error '..i)
end
--]]
