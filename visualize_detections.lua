function visualize_detections(im,boxes,scores,thresh)
  local ok = pcall(require,'qt')
  if not ok then
    error('You need to run visualize_detections using qlua')
  end
  require 'qttorch'
  require 'qtwidget'

  -- select best scoring boxes without background
  local max_score,idx = scores[{{},{2,-1}}]:max(2)

  local idx_thresh = max_score:gt(thresh)
  max_score = max_score[idx_thresh]
  idx = idx[idx_thresh]

  local r = torch.range(1,boxes:size(1)):long()
  local rr = r[idx_thresh]
  local boxes_thresh = boxes:index(1,rr)

  local num_boxes = boxes_thresh:size(1)
  local widths  = boxes_thresh[{{},3}] - boxes_thresh[{{},1}]
  local heights = boxes_thresh[{{},4}] - boxes_thresh[{{},2}]

  local x,y = im:size(3),im:size(2)
  local w = qtwidget.newwindow(x,y,"test")
  local qtimg = qt.QImage.fromTensor(im)
  w:image(0,0,x,y,qtimg)
  local fontsize = 10

  for i=1,num_boxes do
    local x,y = boxes_thresh[{i,1}],boxes_thresh[{i,2}]
    local width,height = widths[i], heights[i]
    
    -- add bbox
    w:rectangle(x,y,width,height)
    
    -- add score
    w:moveto(x,y+fontsize)
    w:setcolor("red")
    w:setfont(qt.QFont{serif=true,italic=true,size=fontsize,bold=true})
    w:show(string.format('%d: %.2f',idx[i],max_score[i]))
  end
  w:setcolor("red")
  w:setlinewidth(2)
  w:stroke()
end
