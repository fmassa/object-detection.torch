
local configs = {}

local image_transformer_params = {
  mean_pix={102.9801,115.9465,122.7717},
  raw_scale = 255,
  swap = {3,2,1}
}

configs.image_transformer_params = image_transformer_params
configs.algo = {}

--------------------------------------------------------------------------------
-- RCNN
--------------------------------------------------------------------------------

local fp_params = {
  crop_size         = 227,
  padding           = 16,
  use_square        = false,
  image_transformer = image_transformer
}
local bp_params = {
  iter_per_batch = 100,
  nTimesMoreData = 10,
  batch_size = opt.batch_size,
  fg_fraction = opt.fg_frac,
  fg_threshold = 0.5,
  bg_threshold = {0.0,0.5},
  do_flip = true,
--  batch_dim = {3,fp_params.crop_size,fp_params.crop_size},
}

local RCNN = {
  fp_params=fp_params,
  bp_params=bp_params,
  bp = nnf.BatchProvider
}

configs.algo.RCNN = RCNN

--------------------------------------------------------------------------------
-- SPP
--------------------------------------------------------------------------------
--
local num_chns = 256
local pooling_scales = {{1,1},{2,2},{3,3},{6,6}}
local pyr = torch.Tensor(pooling_scales):t()
local pooled_size = pyr[1]:dot(pyr[2])
local feat_dim = {num_chns*pooled_size}

local fp_params = {
  scales            = {480,576,688,874,1200},
  randomscale       = true,
  sz_conv_standard  = 13,
  step_standard     = 16,
  offset0           = 21,
  offset            = 6.5,
  inputArea         = 224^2,
  pooling_scales    = pooling_scales,
  num_feat_chns     = num_chns,
  image_transformer = image_transformer
}
local bp_params = {
  iter_per_batch = 500,
  nTimesMoreData = 10,
  batch_size = opt.batch_size,
  fg_fraction = opt.fg_frac,
  fg_threshold = 0.5,
  bg_threshold = {0.1,0.5},
  do_flip = true,
--  batch_dim = feat_dim,
}

local SPP = {
  fp_params=fp_params,
  bp_params=bp_params,
  bp = nnf.BatchProvider
}

configs.algo.SPP = SPP

--------------------------------------------------------------------------------
-- Fast-RCNN
--------------------------------------------------------------------------------

local fp_params = {
  scale             = {600},
  max_size          = 1000,
  image_transformer = image_transformer
}
local bp_params = {
  imgs_per_batch = 2,
  batch_size = opt.batch_size,
  fg_fraction = opt.fg_frac,
  fg_threshold = 0.5,
  bg_threshold = {0.0,0.5},
  do_flip = true,
}

local FRCNN = {
  fp_params=fp_params,
  bp_params=bp_params,
  bp = nnf.BatchProviderROI
}

configs.algo.FRCNN = FRCNN


return configs
