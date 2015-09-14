--------------------------------------------------------------------------------
-- Prepare data model
--------------------------------------------------------------------------------
paths.mkdir(opt.save)

trainCache = paths.concat(opt.save_base,'trainCache.t7')
testCache = paths.concat(opt.save_base,'testCache.t7')

local pooler
local feat_dim

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

image_transformer = nnf.ImageTransformer{mean_pix=image_mean}

if paths.filep(trainCache) then
  print('Loading train metadata from cache')
  batch_provider = torch.load(trainCache)
  feat_provider = batch_provider.feat_provider
  ds_train = feat_provider.dataset
  feat_provider.model = features
else
  ds_train = nnf.DataSetPascal{image_set='trainval',classes=classes,year=opt.year,
                         datadir=opt.datadir,roidbdir=opt.roidbdir}
  
  if opt.algo == 'SPP' then
    feat_provider = nnf.SPP(ds_train)-- remove features here to reduce cache size
    feat_provider.cachedir = paths.concat(opt.cache,'features',opt.netType)
    feat_provider.randomscale = true
    feat_provider.scales = {600}
    feat_provider.spp_pooler = pooler:clone()
    feat_provider.image_transformer = image_transformer
  elseif opt.algo == 'RCNN' then
    feat_provider = nnf.RCNN(ds_train)
    feat_provider.crop_size = feat_dim[2]
    feat_provider.image_transformer = image_transformer
  else
    error(("Detection framework '%s' not available"):format(opt.algo))
  end
  
  print('==> Preparing BatchProvider for training')
  batch_provider = nnf.BatchProvider(feat_provider)
  batch_provider.iter_per_batch = opt.ipb
  batch_provider.nTimesMoreData = opt.ntmd
  batch_provider.batch_size = opt.batch_size
  batch_provider.fg_fraction = opt.fg_frac
  batch_provider.bg_threshold = {0.0,0.5}
  batch_provider.do_flip = true
  batch_provider.batch_dim = feat_dim
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
  if opt.algo == 'SPP' then
    feat_provider_test = nnf.SPP(ds_test)
    feat_provider_test.randomscale = false
    feat_provider_test.cachedir = paths.concat(opt.cache,'features',opt.netType)
    feat_provider_test.scales = {600}
    feat_provider_test.spp_pooler = pooler:clone()
    feat_provider_test.image_transformer = image_transformer
  elseif opt.algo == 'RCNN' then
    feat_provider_test = nnf.RCNN(ds_test)
    feat_provider_test.crop_size = feat_dim[2]
    feat_provider_test.image_transformer = image_transformer
  else
    error(("Detection framework '%s' not available"):format(opt.algo))
  end
  
  print('==> Preparing BatchProvider for validation')
  batch_provider_test = nnf.BatchProvider(feat_provider_test)
  batch_provider_test.iter_per_batch = 500--opt.ipb
  batch_provider_test.nTimesMoreData = 10--opt.ntmd
  batch_provider_test.batch_size = opt.batch_size
  batch_provider_test.fg_fraction = opt.fg_frac
  batch_provider_test.bg_threshold = {0.0,0.5}
  batch_provider_test.do_flip = false
  batch_provider_test.batch_dim = feat_dim
  batch_provider_test:setupData()
  
  torch.save(testCache,batch_provider_test)
  feat_provider_test.model = features
end

-- compute feature cache

features = nil
model = nil

collectgarbage()
