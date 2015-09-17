--------------------------------------------------------------------------------
-- Prepare data model
--------------------------------------------------------------------------------
paths.mkdir(opt.save)

trainCache = paths.concat(opt.save_base,'trainCache.t7')
testCache = paths.concat(opt.save_base,'testCache.t7')

local pooler
local feat_dim
--[[
if opt.algo == 'SPP' then
  local conv_list = features:findModules(opt.backend..'.SpatialConvolution')
  local num_chns = conv_list[#conv_list].nOutputPlane
  pooler = model:get(2):clone():float()
  local pyr = torch.Tensor(pooler.pyr):t()
  local pooled_size = pyr[1]:dot(pyr[2])
  feat_dim = {num_chns*pooled_size}
elseif opt.algo == 'RCNN' then
  feat_dim = {3,227,227}
end
--]]

image_transformer = nnf.ImageTransformer{mean_pix=image_mean}


local FP        = nnf[opt.algo]
local fp_params = config.algo[opt.algo].fp_params
local bp_params = config.algo[opt.algo].bp_params
local BP        = config.algo[opt.algo].bp

if paths.filep(trainCache) then
  print('Loading train metadata from cache')
  batch_provider = torch.load(trainCache)
  feat_provider = batch_provider.feat_provider
  ds_train = feat_provider.dataset
  feat_provider.model = features
else
  ds_train = nnf.DataSetPascal{image_set='trainval',classes=classes,year=opt.year,
                         datadir=opt.datadir,roidbdir=opt.roidbdir}
  

  feat_provider = FP(ds_train)
  batch_provider = BP(bp_params)
  batch_provider:setupData()

  torch.save(trainCache,batch_provider)
  feat_provider.model = features
end

if paths.filep(testCache) then
  print('Loading test metadata from cache')
  batch_provider_test = torch.load(testCache)
  feat_provider_test = batch_provider_test.feat_provider
  ds_test = feat_provider_test.dataset
  feat_provider_test.model = features
else
  ds_test = nnf.DataSetPascal{image_set='test',classes=classes,year=opt.year,
                              datadir=opt.datadir,roidbdir=opt.roidbdir}


  feat_provider_test = FP(ds_test)
  -- disable flip ?
  bp_params.do_flip = false
  batch_provider_test = BP(bp_params)

  batch_provider_test:setupData()
  
  torch.save(testCache,batch_provider_test)
  feat_provider_test.model = features
end

-- compute feature cache

features = nil
model = nil

collectgarbage()
