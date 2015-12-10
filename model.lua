require 'nn'
--require 'inn'
--require 'cudnn'

local createModel = paths.dofile('models/' .. opt.netType .. '.lua')
print('=> Creating model from file: models/' .. opt.netType .. '.lua')
local model = createModel()

local criterion = nn.CrossEntropyCriterion()

print('Model:')
print(model)
print('Criterion:')
print(criterion)

-- If preloading option is set, preload weights from existing models appropriately
if opt.retrain ~= 'none' then
  assert(paths.filep(opt.retrain), 'File not found: ' .. opt.retrain)
  print('Loading model from file: ' .. opt.retrain);
  model = torch.load(opt.retrain)
end

collectgarbage()

return model, criterion

