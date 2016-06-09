require 'nn'
require 'image'
--require 'inn'
require 'xlua'

local objdet = require 'objdet.env'
--nnf = {}

--torch.include('nnf','ImageTransformer.lua')
require 'objdet.ImageTransformer'

--torch.include('nnf','DataSetDetection.lua')
--torch.include('nnf','DataSetPascal.lua')
--torch.include('nnf','DataSetCOCO.lua')
require 'objdet.DataSetDetection'
require 'objdet.DataSetPascal'
require 'objdet.DataSetCOCO'

--torch.include('nnf','BatchProviderBase.lua')
--torch.include('nnf','BatchProviderIC.lua')
--torch.include('nnf','BatchProviderRC.lua')
require 'objdet.BatchProviderBase'
require 'objdet.BatchProviderIC'
require 'objdet.BatchProviderRC'

--torch.include('nnf','SPP.lua')
--torch.include('nnf','RCNN.lua')
--torch.include('nnf','FRCNN.lua')
--torch.include('nnf','ROIPooling.lua')
require 'objdet.SPP'
require 'objdet.RCNN'
require 'objdet.FRCNN'

require 'objdet.ROIPooling'

--torch.include('nnf','Trainer.lua')
--torch.include('nnf','Tester.lua')
--require 'objdet.Trainer'
--require 'objdet.Tester'

--torch.include('nnf','SVMTrainer.lua')
--require 'objdet.SVMTrainer'

--torch.include('nnf','ImageDetect.lua')
require 'objdet.ImageDetect'
--return nnf
return objdet
