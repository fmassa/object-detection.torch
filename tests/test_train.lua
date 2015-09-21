dofile 'tests/test_utils.lua'

--------------------------------------------------------------------------------
-- define batch providers
--------------------------------------------------------------------------------

bp1 = nnf.BatchProvider{dataset=ds,feat_provider=fp1}
bp1.nTimesMoreData = 2
bp1.iter_per_batch = 10
bp2 = nnf.BatchProviderROI{dataset=ds,feat_provider=fp2}

bp1.bboxes = torch.load('tests/bproibox.t7')
bp2.bboxes = torch.load('tests/bproibox.t7')

--------------------------------------------------------------------------------
--
--------------------------------------------------------------------------------

criterion = nn.CrossEntropyCriterion()

trainer = nnf.Trainer(model1,criterion,bp1)

for i=1,10 do
  trainer:train(10)
end

