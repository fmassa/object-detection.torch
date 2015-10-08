dofile 'tests/test_utils.lua'
I = ds:getImage(1)
boxes = ds:getROIBoxes(1)
scores = torch.rand(boxes:size(1),21)
dofile 'visualize_detections.lua' 
visualize_detections(I,boxes,scores,0.9)

