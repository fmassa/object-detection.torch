local SVMTrainer = torch.class('nnf.SVMTrainer')

function SVMTrainer:__init(module,feat_provider)
  --self.dataset = dataset
  self.module = module
  self.feat_provider = feat_provider

  self.feat_dim = {256*50}
  self.batch_size = 128
  self.max_batch_size = 15000 

  self.negative_overlap = {0,0.3}

  self.first_time = true

  self.svm_C = 1e-3
  self.bias_mult = 10
  self.pos_loss_weight = 2

  self.retrain_limit = 2000
  self.evict_thresh = -1.2
  self.hard_thresh = -1.0001

  self.pos_feat_type = 'real' -- real, mixed, synthetic
 
  self.synth_neg = true

  --self:getFeatureStats()
end


function SVMTrainer:getFeatureStats(dataset,feat_provider,module)

  if false then
    self.mean_norm = 19.848824140978--30.578503376687
    return
  end

  local feat_provider = feat_provider or self.feat_provider
  local module = module or self.module
  local dataset = dataset

  local boxes_per_image = 200
  local num_images = math.min(dataset:size(),200)

  local valid_idx = torch.randperm(dataset:size())
  valid_idx = valid_idx[{{1,num_images}}]

  local feat_cumsum = 0
  local feat_n = 0
  local bboxes = torch.IntTensor(boxes_per_image,4)
  
  print('Getting feature stats')
  for i=1,num_images do
    xlua.progress(i,num_images)
    local img_idx = valid_idx[i]
    local I = dataset:getImage(img_idx)
    local rec = dataset:attachProposals(img_idx)
    
    local num_bbox = math.min(boxes_per_image,rec:size())

    local bbox_idx = torch.randperm(rec:size()):long()
    bbox_idx = bbox_idx[{{1,num_bbox}}]

    bboxes:index(rec.boxes,1,bbox_idx)
    
    local feat = feat_provider:getFeature(I,bboxes)
    local final_feat = feat_provider:compute(module, feat)

    feat_n = feat_n + num_bbox
    feat_cumsum = feat_cumsum + final_feat:pow(2):sum(2):sqrt():sum()
  end
  self.mean_norm = feat_cumsum/feat_n
end

function SVMTrainer:scaleFeatures(feat)
  local target_norm = 20
  feat:mul(target_norm/self.mean_norm)
end

function SVMTrainer:getPositiveFeatures(dataset,feat_provider,module)
  local feat_provider = feat_provider or self.feat_provider
  local module = module or self.module
  local dataset = dataset
  module:evaluate()
  local positive_data = {}
  for cl_idx,cl_name in pairs(dataset.classes) do
    positive_data[cl_name] = {}
  end
  local fc5_feat = torch.FloatTensor()
  local fc7_feat = torch.FloatTensor()
  local fc7_idxs = torch.linspace(1,4096,4096):int()
  local end_idx = dataset:size()
  local not_done = torch.ByteTensor(dataset.num_classes):fill(1)
  for i=1,end_idx do
    xlua.progress(i,end_idx)
    local I = dataset:getImage(i)
    --local gt_boxes, gt_classes = dataset:getGTBoxes(i)



    local rec = dataset:attachProposals(i)
    local overlap = rec.overlap_class
    local is_gt = rec.gt

    for cl_idx,cl_name in pairs(dataset.classes) do
      if overlap:numel()>0 then
        local num_pos = overlap[{{},cl_idx}]:eq(1):float():dot(is_gt:float())
        fc5_feat:resize(num_pos,unpack(self.feat_dim))
        fc7_feat:resize(num_pos,4096)
        local count = 0
        for j=1,rec:size() do
          if overlap[j][cl_idx]==1 and is_gt[j]==1 then
            count = count + 1
            local fff = feat_provider:getFeature(I,rec.boxes[j])[1]
            --print(fff:size())
            --print(fc5_feat:size())
            fc5_feat[count] = fff
          end
        end
        if num_pos > 0 then
          fc7_feat:copy(module:forward(fc5_feat:cuda()))
          self:scaleFeatures(fc7_feat)
        end
        for j=1,num_pos do
          local f = fc7_feat[j]
          if j==1 and f[4096]==0 and not_done[cl_idx] == 1 then
            table.insert(positive_data[cl_name],{1,{fc7_idxs:clone(),f:clone()}})
            not_done[cl_idx] = 0
          else
            table.insert(positive_data[cl_name],{1,{fc7_idxs[f:ne(0)],f[f:ne(0)]}})
          end
        end
      end
    end
  end
  return positive_data
end

function SVMTrainer:sampleNegativeFeatures(ind,dataset,feat_provider,module)

  local feat_provider = feat_provider or self.feat_provider
  local dataset = dataset
  local module = module or self.module
  module:evaluate()
collectgarbage()
  local first_time = self.first_time

  local I = dataset:getImage(ind)
  local rec = dataset:attachProposals(ind)
  local overlap = rec.overlap_class

  local fc5_feat = torch.FloatTensor()
  local fc7_feat = torch.FloatTensor()
  local fc7_idxs = torch.linspace(1,4096,4096):int()

  local caches = {}
  for cl_idx,cl_name in pairs(dataset.classes) do
    caches[cl_name] = {X_neg = {},num_added = 0}
  end

  local feat = feat_provider:getFeature(I,rec.boxes)
  local fc7_feat = feat_provider:compute(module, feat)

  self:scaleFeatures(fc7_feat)

  if first_time then
    for cl_idx,cl_name in pairs(dataset.classes) do
      local count = 0
      local nsize = 0
      for j=1,rec:size() do
        if overlap[j][cl_idx] >= self.negative_overlap[1] and
           overlap[j][cl_idx] <  self.negative_overlap[2] then
          local f = fc7_feat[j]
          table.insert(caches[cl_name].X_neg,{-1,{fc7_idxs[f:ne(0)],f[f:ne(0)]}})
          caches[cl_name].num_added = caches[cl_name].num_added + 1
        end
      end  
    end

    self.first_time = false

  else 

    local W = self.W
    local B = self.B:view(dataset.num_classes,1):expand(dataset.num_classes,fc7_feat:size(1))

    local zs = torch.addmm(B:float(),W:float(),fc7_feat:t())
    
    for cl_idx,cl_name in pairs(dataset.classes) do
      local z = zs[cl_idx]
      for j=1,rec:size() do
        if z[j] > self.hard_thresh and
           overlap[j][cl_idx] >= self.negative_overlap[1] and
           overlap[j][cl_idx] <  self.negative_overlap[2] then
          local f = fc7_feat[j]
          table.insert(caches[cl_name].X_neg,{-1,{fc7_idxs[f:ne(0)],f[f:ne(0)]}})
          caches[cl_name].num_added = caches[cl_name].num_added + 1
        end
      end
    end
  end
  return caches
end

local function mergeTables(pos,neg,inplace)
  if not inplace then
    local res = {}
    for k,v in pairs(pos) do
      res[k] = v
    end
    local npos = #pos
    for k,v in pairs(neg) do
      res[npos+k] = v
    end
    return res
  else
    local nneg = #neg
    for k,v in pairs(pos) do
      neg[nneg+k] = v
    end
  end
end

local function sparse2full(res,v,idx)
  res:zero()
  local res_data = torch.data(res)
  local s_num = v:size(1)
  local s_idx = torch.data(idx)
  local s_val = torch.data(v)
  for jj=1,s_num do
    res_data[s_idx[jj-1] -1 ] = s_val[jj-1]
  end

end

function SVMTrainer:selectPositiveFeatures()
  if self.pos_feat_type == 'real' then
    self.positive_data = self:getPositiveFeatures()
  elseif self.pos_feat_type == 'synthetic' then
    self.positive_data = self:getPositiveFeatures(self.feat_provider_synth,self.module_synth)
  elseif self.pos_feat_type == 'mixed' then
    self.positive_data = self:getPositiveFeatures()
    local X_pos_synth = self:getPositiveFeatures(self.feat_provider_synth,self.module_synth)
    for cl_name,feat_val in pairs(X_pos_synth) do
      mergeTables(feat_val,self.positive_data[cl_name],true)
    end
  else
    error('Mixture type not supported!')
  end
end

function SVMTrainer:setPositiveDataType(pos_feat_type,feat_provider_synth,module_synth)
  self.pos_feat_type = pos_feat_type
  self.feat_provider_synth = feat_provider_synth
  self.module_synth = module_synth
end

function SVMTrainer:addPositiveFeatures(feat_provider,module)
  local X_pos = self:getPositiveFeatures(feat_provider,module)
  for cl_name,feat_val in pairs(X_pos) do
    if not self.positive_data[cl_name] then
      self.positive_data[cl_name] = {}
    end
    mergeTables(feat_val,self.positive_data[cl_name],true)
  end
end


function SVMTrainer:train(dataset)
  --local dataset = self.dataset
  
  --print('Experiment name: '..self.expname)

  self.W = torch.Tensor(dataset.num_classes,4096)
  self.B = torch.Tensor(dataset.num_classes)

  --self:selectPositiveFeatures()
  --self:addPositiveFeatures()
  
  local caches = {}
  for cl_idx,cl_name in pairs(dataset.classes) do
    caches[cl_name] = {X_neg = {},num_added = 0,X_neg_num=0,
                       pos_loss = {}, neg_loss = {},
                       reg_loss = {}, tot_loss = {}}
  end
  local X_all
  local first_time = true

  local liblinear_type = 3
  local svm_params = '-w1 '..self.pos_loss_weight..
                     ' -c '..self.svm_C..
                     ' -s '..liblinear_type..
                     ' -B '..self.bias_mult..
                     ' -q'

  print('svm parameters: '..svm_params)
  local end_iter = dataset:size()
  self.svm_model = {}

  local has_synth = false
  local num_synth = 0
  if self.feat_provider_synth and self.synth_neg then
    num_synth = self.feat_provider_synth.dataset:size()
    has_synth = true
    end_iter = end_iter + num_synth
  end

  for i=1,end_iter do
    print('hard neg epoch: image '..i..'/'..end_iter)

    if has_synth and self.synth_neg then
      if i<= num_synth then
        X = self:sampleNegativeFeatures(i,self.feat_provider_synth,self.module_synth)
      else
        X = self:sampleNegativeFeatures(i-num_synth)
      end
    else
      X = self:sampleNegativeFeatures(i,dataset)
    end

    for cl_idx,cl_name in pairs(dataset.classes) do
      local timer = torch.Timer()
      if X[cl_name].num_added > 0 then
        mergeTables(X[cl_name].X_neg,caches[cl_name].X_neg,true)
        caches[cl_name].X_neg_num = caches[cl_name].X_neg_num + X[cl_name].num_added
        caches[cl_name].num_added = caches[cl_name].num_added + X[cl_name].num_added
      end
      local is_last_time = (i == end_iter)
      local hit_retrain_limit = caches[cl_name].num_added > self.retrain_limit
      if (first_time or hit_retrain_limit or is_last_time) and caches[cl_name].X_neg_num > 0 then
        print('>>>Updating '..cl_name..' detector<<<')
        print('Cache holds '..#self.positive_data[cl_name]..' pos examples '..
              #caches[cl_name].X_neg..' neg examples')
  
        X_all = mergeTables(self.positive_data[cl_name],caches[cl_name].X_neg,false)
        m = liblinear.train(X_all,svm_params)
        self.W[cl_idx] = m.weight[{1,{1,4096}}]
        self.B[cl_idx] = m.weight[{1,4097}]*self.bias_mult

        self.svm_model[cl_idx] = m

        caches[cl_name].num_added = 0

        local W = self.W[cl_idx]:float()
        local B = self.B[cl_idx]
        
        local z_pos = torch.FloatTensor(#self.positive_data[cl_name]):zero()
        local z_neg = torch.FloatTensor(#caches[cl_name].X_neg):zero()
        
        local fc7_feat = torch.FloatTensor(4096)

        for el_idx,el in pairs(self.positive_data[cl_name]) do
          sparse2full(fc7_feat,el[2][2],el[2][1])
--          assert(fc7_feat[fc7_feat:ne(0)]:eq(el[2][2]):all())
          z_pos[el_idx] = fc7_feat:dot(W) + B
        end

        local easy = {}
        for el_idx,el in pairs(caches[cl_name].X_neg) do
          sparse2full(fc7_feat,el[2][2],el[2][1])
--          assert(fc7_feat[fc7_feat:ne(0)]:eq(el[2][2]):all())
          z_neg[el_idx] = fc7_feat:dot(W) + B
          if z_neg[el_idx] < self.evict_thresh then
            table.insert(easy,el_idx)
          end
        end
        -- remove easy ones
        for jj=#easy,1,-1 do
          table.remove(caches[cl_name].X_neg,easy[jj])
        end
        caches[cl_name].X_neg_num = caches[cl_name].X_neg_num - #easy

        local pos_loss = self.svm_C * self.pos_loss_weight * 
                         z_pos:mul(-1):add(1):clamp(0,math.huge):sum()

        local neg_loss = self.svm_C * z_neg:add(1):clamp(0,math.huge):sum()

        local reg_loss = 0.5 * W:dot(W) + 0.5 * (B / self.bias_mult)^2;
        local tot_loss = pos_loss + neg_loss + reg_loss

        table.insert(caches[cl_name].pos_loss,pos_loss)
        table.insert(caches[cl_name].neg_loss,neg_loss)
        table.insert(caches[cl_name].reg_loss,reg_loss)
        table.insert(caches[cl_name].tot_loss,tot_loss)

        local cc = caches[cl_name]
        for t=1,#caches[cl_name].tot_loss do
          local ss = string.format('   %2d: obj val: %.3f = %.3f (pos) + %.3f (neg) + %.3f (reg)',t,cc.tot_loss[t],cc.pos_loss[t],cc.neg_loss[t],cc.reg_loss[t])
          print(ss)
        end

        print('  Prunning '.. #easy ..' easy negatives')
        print('  Cache holds '..#self.positive_data[cl_name].. ' pos examples '..
              #caches[cl_name].X_neg..' neg examples')

        print('Elapsed time: '..timer:time().real..' s')
      end 
    end
    first_time = false
  end
  --torch.save('/home/francisco/work/projects/cross_domain/cachedir/svm_models/svm_model,'..self.expname..'.t7',{W=self.W,B=self.B})
  return caches--X_all
end

function SVMTrainer:test(feat_provider_test)

  local feat_provider = feat_provider_test
  local dataset = feat_provider.dataset
  local module = self.module
  --local batch_size = self.batch_size
  self.cachefolder = '/home/francisco/work/projects/cross_domain/cachedir/results_svm/svm,'..self.expname
  local pathfolder = paths.concat(self.cachefolder,'test')
  paths.mkdir(pathfolder)  

  module:evaluate()
  dataset:loadROIDB()
  
  local fc5_feat = torch.Tensor():float()
  local fc7_feat = torch.Tensor():float()
  
  local W = self.W
  local B = self.B:view(dataset.num_classes,1)--:expand(dataset.num_classes,fc7_feat:size(1))

  local output = torch.FloatTensor()
  local boxes
  
  for i=1,dataset:size() do
    xlua.progress(i,dataset:size())
    boxes = dataset.roidb[i]
    local num_boxes = boxes:size(1)
    
    --local batch_size = num_boxes > self.max_batch_size and self.batch_size or num_boxes
    --local num_batches = math.ceil(num_boxes/batch_size)
    --local batch_rest = num_boxes%batch_size
    
    --feats:resize(batch_size,unpack(feat_dim))
    fc5_feat:resize(num_boxes,unpack(self.feat_dim))
    for idx=1,num_boxes do
      fc5_feat[idx] = feat_provider:getFeature(i,boxes[idx])
    end
--    output = module:forward(feats:cuda())
    fc7_feat:resize(num_boxes,4096):copy(module:forward(fc5_feat:cuda()))
    self:scaleFeatures(fc7_feat)

    B = self.B:view(dataset.num_classes,1):expand(dataset.num_classes,num_boxes)
--print(fc7_feat:size())
--print(B:size())
--print(W:size())
    output:resize(dataset.num_classes,num_boxes)
    output:addmm(B:float(),W:float(),fc7_feat:t())

    output = output:t()
    --[[ make more general later, not in the mood
    for b = 1,num_batches-1 do
    
      for idx=1,batch_size do
        feats[idx] = feat_provider:getFeature(i,boxes[(b-1)*batch_size + idx])
      end
      
      output = module:forward(feats)
      
    end]]
    collectgarbage()
    --torch.save(paths.concat(self.cachefolder,module.experiment,))
    mattorch.save(paths.concat(pathfolder,dataset.img_ids[i]..'.mat'),output)
  end
  
  -- clean roidb to free memory
  dataset.roidb = nil
end



