require 'nn'
nnf = {}
dofile 'ROIPooling.lua'

m = nnf.ROIPooling(3,3)

t = {torch.rand(1,10,10),torch.Tensor({{1,1,5,5},{2,3,7,8},{6,4,8,8},{6,4,10,10},{8,8,10,10}})} -- 
g = torch.rand(t[2]:size(1),1,3,3)

o = m:forward(t)
gg = m:backward(t,g)


