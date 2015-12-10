trainer = nnf.Trainer(model, criterion, batch_provider)

local num_iter = opt.num_iter/opt.disp_iter
local lr_step = opt.lr_step/opt.disp_iter
local save_step = opt.save_step/opt.disp_iter

trainer.optimState.learningRate = opt.lr

logger = optim.Logger(paths.concat(rundir,'train.log'))

if opt.conf_mat then
  local conf_classes = {'background'}
  for k,v in ipairs(ds_train.classes) do
    table.insert(conf_classes,v)
  end
  trainer.confusion = optim.ConfusionMatrix(conf_classes)
end

local lightModel = model:clone('weight','bias','running_mean','running_std')

-- main training loop
for i=1,num_iter do
  if i % lr_step == 0 then
    trainer.optimState.learningRate = trainer.optimState.learningRate/10
  end
  print(('Iteration %3d/%-3d'):format(i,num_iter))
  trainer:train(opt.disp_iter)
  print(('  Training error: %.5f'):format(trainer.fx[i]))

  if opt.conf_mat then
    print(trainer.confusion)
    logger:add{
      ['train error']=trainer.fx[i],
      ['confusion matrix']=tostring(trainer.confusion),
      ['learning rate']=trainer.optimState.learningRate
    }
  else
    logger:add{
      ['train error']=trainer.fx[i],
      ['learning rate']=trainer.optimState.learningRate
    }
  end

  if i% save_step == 0 then
    torch.save(paths.concat(rundir, 'model.t7'), lightModel)
  end
end

torch.save(paths.concat(rundir, 'model.t7'), lightModel)
