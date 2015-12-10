local M = {}

function M.parse(arg)

  local cmd = torch.CmdLine()
  cmd:text()
  cmd:text('Object detection in torch')
  cmd:text()
  cmd:text('Options:')

  cmd:option('-name',      'obj-detect',   'base name')
  cmd:option('-algo',      'RCNN',         'Detection framework. Options: RCNN | FRCNN')
  cmd:option('-netType',   'alexnet',      'Options: alexnet')
  cmd:option('-lr',        1e-3,           'learning rate')
  cmd:option('-num_iter',  40000,          'number of iterations')
  cmd:option('-disp_iter', 100,            'display every n iterations')
  cmd:option('-lr_step',   30000,          'step for reducing the learning rate')
  cmd:option('-save_step', 10000,          'step for saving the model')
  cmd:option('-gpu',       1,              'gpu to use (0 for cpu mode)')
  cmd:option('-conf_mat',  false,          'Compute confusion matrix during training')
  cmd:option('-seed',      1,              'fix random seed (if ~= 0)')
  cmd:option('-numthreads',6,              'number of threads')
  cmd:option('-retrain',   'none',         'modelpath for finetuning')

  local opt = cmd:parse(arg or {})

  local exp_name = cmd:string(opt.name, opt, {name=true, gpu=true, numthreads=true})

  rundir = 'cachedir/'..exp_name
  paths.mkdir(rundir)

  cmd:log(paths.concat(rundir,'log'), opt)
  cmd:addTime('Object-Detection.Torch')

  return opt

end

return M
