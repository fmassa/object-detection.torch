require 'nn'
require 'optim'
require 'xlua'

local Trainer = torch.class('nnf.Trainer')

function Trainer:__init(module,criterion)
  
  self.module = module
  self.criterion = criterion
  
  self.parameters,self.gradParameters = self.module:getParameters()
  
  self.optimState = {learningRate = 1e-3, weightDecay = 0.0005, momentum = 0.9,
                     learningRateDecay = 0}
                     
  self.epoch = 0

  self.normalize = false  

  self.fx = {}
  
end


function Trainer:train(inputs,targets)
  -- only for batches
  assert(targets:dim()>2,'Trainer is only for batches')
  
  self.module:training()
  self._input = self._input or torch.CudaTensor()
  self._target = self._target or torch.CudaTensor()

  local module = self.module
  local parameters = self.parameters
  local gradParameters = self.gradParameters
  
  local criterion = self.criterion
  local optimState = self.optimState
    
  local batchSize = inputs:size(2)
  local maxIter = inputs:size(1)
  
  if self.confusion then
    self.confusion:zero()
  end
  local err = 0
  
  self._input:resize(inputs[1]:size())
  self._target:resize(targets[1]:size())
  local input = self._input
  local target = self._target
  
  for t=1,maxIter do
    xlua.progress(t,maxIter)

    input:copy(inputs[t])
    target:copy(targets[t])

    local feval = function(x)
      if x ~= parameters then
        parameters:copy(x)
      end
      gradParameters:zero()
      
      local outputs = module:forward(input)
      
      local f = criterion:forward(outputs,target)
      local df_do = criterion:backward(outputs,target)
      
      module:backward(input,df_do)
      
      if self.normalize then
        gradParameters:div(batchSize)
        f = f/batchSize
      end

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
