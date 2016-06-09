require 'nn'
require 'image'
--require 'inn'
require 'xlua'

local objdet = require 'objdet.env'

require 'objdet.ImageTransformer'

require 'objdet.DataSetDetection'
require 'objdet.DataSetPascal'
require 'objdet.DataSetCOCO'

require 'objdet.BatchProviderBase'
require 'objdet.BatchProviderIC'
require 'objdet.BatchProviderRC'

require 'objdet.SPP'
require 'objdet.RCNN'
require 'objdet.FRCNN'

require 'objdet.ROIPooling'

--torch.include('nnf','Trainer.lua')
--torch.include('nnf','Tester.lua')
require 'objdet.Trainer'
require 'objdet.Tester'

--torch.include('nnf','SVMTrainer.lua')
require 'objdet.SVMTrainer'

--torch.include('nnf','ImageDetect.lua')
require 'objdet.ImageDetect'
--return nnf
return objdet
