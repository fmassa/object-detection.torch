--------------------------------------------------------------------------------
-- Compute conv5 feature cache (for SPP)
--------------------------------------------------------------------------------
if opt.algo == 'SPP' then
  print('Preparing conv5 features for '..ds_train.dataset_name..' '
        ..ds_train.image_set)
  local feat_cachedir = feat_provider.cachedir
  for i=1,ds_train:size() do
    xlua.progress(i,ds_train:size())
    local im_name = ds_train.img_ids[i]
    local cachefile = paths.concat(feat_cachedir,im_name)
    if not paths.filep(cachefile..'.h5') then
      local f = feat_provider:getConv5(i)
    end
    if not paths.filep(cachefile..'_flip.h5') then
      local f = feat_provider:getConv5(i,true)
    end
    if i%50 == 0 then
      collectgarbage()
      collectgarbage()
    end
  end
  
  print('Preparing conv5 features for '..ds_test.dataset_name..' '
        ..ds_test.image_set)
  local feat_cachedir = feat_provider_test.cachedir
  for i=1,ds_test:size() do
    xlua.progress(i,ds_test:size())
    local im_name = ds_test.img_ids[i]
    local cachefile = paths.concat(feat_cachedir,im_name)
    if not paths.filep(cachefile..'.h5') then
      local f = feat_provider_test:getConv5(i)
    end
    if i%50 == 0 then
      collectgarbage()
      collectgarbage()
    end
  end
end

