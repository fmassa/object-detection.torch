--------------------------------------------------------------------------------
-- Prepare data model
--------------------------------------------------------------------------------

local trainCache = paths.concat(rundir,'trainCache.t7')
--testCache = paths.concat(opt.save_base,'testCache.t7')

local config = paths.dofile('config.lua')

image_transformer = nnf.ImageTransformer(config.image_transformer_params)

local FP        = nnf[opt.algo]
local fp_params = config.algo[opt.algo].fp_params
local bp_params = config.algo[opt.algo].bp_params
local BP        = config.algo[opt.algo].bp

local train_params = config.train_params

-- add common parameters
fp_params.image_transformer = image_transformer
for k,v in pairs(train_params) do
  bp_params[k] = v
end

-------------------------------------------------------------------------------
-- Create structures
--------------------------------------------------------------------------------

ds_train = nnf.DataSetPascal{
  image_set='trainval',
  year=2007,--opt.year,
  datadir=config.datasetDir,
  roidbdir=config.roidbDir
}

feat_provider = FP(fp_params)
feat_provider:training()

bp_params.dataset = ds_train
bp_params.feat_provider = feat_provider
batch_provider = BP(bp_params)

if paths.filep(trainCache) then
  print('Loading train metadata from cache')
  local metadata = torch.load(trainCache)
  batch_provider.bboxes = metadata
else
  batch_provider:setupData()
  torch.save(trainCache, batch_provider.bboxes)
end

-- test
ds_test = nnf.DataSetPascal{
  image_set='test',
  year=2007,--opt.year,
  datadir=config.datasetDir,
  roidbdir=config.roidbDir
}

-- only needed because of SPP
-- could be the same as the one for training
--feat_provider_test = FP(fp_params)
--feat_provider_test:evaluate()

collectgarbage()
