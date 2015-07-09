local M = {}

function M.parse(arg)

  local cmd = torch.CmdLine()
  cmd:text()
  cmd:text('Object detection in torch')
  cmd:text()
  cmd:text('Options:')

  local curr_dir = paths.cwd()
  local defaultDataSetDir = paths.concat(curr_dir,'datasets')
  local defaultDataDir = paths.concat(defaultDataSetDir,'VOCdevkit/')
  local defaultROIDBDir = paths.concat(curr_dir,'data','selective_search_data/')
  
  cmd:text('Folder parameters')
  cmd:option('-cache',paths.concat(curr_dir,'cachedir'),'Cache dir')
  cmd:option('-datadir',defaultDataDir,'Path to dataset')
  cmd:option('-roidbdir',defaultROIDBDir,'Path to ROIDB')
  cmd:text()
  cmd:text('Model parameters')
  cmd:option('-algo','SPP','Detection framework. Options: RCNN | SPP')
  cmd:option('-netType','zeiler','Options: zeiler | vgg')
  cmd:option('-backend','cudnn','Options: nn | cudnn')
  cmd:text()
  cmd:text('Data parameters')
  cmd:option('-year',2007,'DataSet year (for Pascal)')
  cmd:option('-ipb',500,'iter per batch')
  cmd:option('-ntmd',10,'nTimesMoreData')
  cmd:option('-fg_frac',0.25,'fg_fraction')
  cmd:option('-classes','all','use all classes (all) or given class')
  cmd:text()
  cmd:text('Training parameters')
  cmd:option('-lr',1e-3,'learning rate')
  cmd:option('-num_iter',300,'number of iterations')
  cmd:option('-nsmooth',10,'number of iterations before reducing learning rate')
  cmd:option('-nred',4,'number of divisions by 2 before stopping learning')
  cmd:option('-nildfdx',false,'erase memory of gradients when reducing learning rate')
  cmd:option('-batch_size',128,'batch size')
  cmd:text()
  cmd:text('Others')
  cmd:option('-gpu',1,'gpu device to use')
  cmd:option('-numthreads',6,'number of threads to use')
  cmd:option('-comment','','additional comment to the name')
  cmd:option('-seed',0,'random seed (0 = no fixed seed)')
  cmd:option('-retrain','none','modelpath for finetuning')
  cmd:text()


  local opt = cmd:parse(arg or {})
  -- add commandline specified options
  opt.save = paths.concat(opt.cache,
                          cmd:string(opt.netType, opt,
                                     {retrain=true, optimState=true, cache=true,
                                      data=true, gpu=true, numthread=true,
                                      netType=true}))
  -- add date/time
  opt.save_base = opt.save
  local date_time = os.date():gsub(' ','')
  opt.save = paths.concat(opt.save, date_time)

  return opt

end

return M
