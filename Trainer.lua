require 'nn'
require 'optim'
require 'xlua'
local utils = paths.dofile('utils.lua')
local recursiveResizeAsCopyTyped = utils.recursiveResizeAsCopyTyped

local Trainer = torch.class('nnf.Trainer')

function Trainer:__init(module,criterion,batch_provider)
  
  self.module = module
  self.criterion = criterion
  self.batch_provider = batch_provider
  
  self.parameters,self.gradParameters = self.module:getParameters()
  
  self.optimState = {learningRate = 1e-3, weightDecay = 0.0005, momentum = 0.9,
                     learningRateDecay = 0}
                     
  self.epoch = 0

  self.normalize = false  

  self.fx = {}
  
end

function Trainer:train()
  
  self.module:training()

  local module = self.module
  local batch_provider = self.batch_provider
  local parameters = self.parameters
  local gradParameters = self.gradParameters
  
  local criterion = self.criterion
  local optimState = self.optimState
    
  --local maxIter = inputs:size(1)
  
  if self.confusion then
    self.confusion:zero()
  end
  local err = 0
  
  local input
  local target

  for t=1,maxIter do
    xlua.progress(t,maxIter)

    -- get training batch
    self.input0,self.target0 = batch_provider(self.input0,self.target0)

    -- copy to ttype
    self.input,self.input0   = recursiveResizeAsCopyTyped(self.input,self.input0,ttype)
    self.target,self.target0 = recursiveResizeAsCopyTyped(self.target,self.target0,ttype)
    input = self.input
    target = self.target

    local feval = function(x)
      if x ~= parameters then
        parameters:copy(x)
      end
      gradParameters:zero()
      
      local outputs = module:forward(input)
      
      local f = criterion:forward(outputs,target)
      local df_do = criterion:backward(outputs,target)
      
      module:backward(input,df_do)
      
      if self.confusion then
        self.confusion:batchAdd(outputs,target)
      end

      return f,gradParameters
    end
    
    local x,fx = optim.sgd(feval,parameters,optimState)
    err = err + fx[1]
  end
  
  table.insert(self.fx,err/maxIter)
  
  self.module:evaluate()
  self.epoch = self.epoch + 1
end
