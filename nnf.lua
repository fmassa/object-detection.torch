require 'nn'
require 'image'
--require 'inn'
require 'xlua'

nnf = {}

torch.include('nnf','DataSetPascal.lua')
torch.include('nnf','BatchProviderBase.lua')
torch.include('nnf','BatchProvider.lua')
torch.include('nnf','BatchProviderROI.lua')

--torch.include('nnf','SPP.lua')
torch.include('nnf','RCNN.lua')
torch.include('nnf','ROIPooling.lua')

torch.include('nnf','Trainer.lua')
torch.include('nnf','Tester.lua')
torch.include('nnf','Tester_FRCNN.lua')

torch.include('nnf','SVMTrainer.lua')

torch.include('nnf','ImageTransformer.lua')
torch.include('nnf','ImageDetect.lua')
--return nnf
