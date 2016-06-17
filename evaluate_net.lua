require 'cudnn'
require 'eval_helper'
require 'image'


function EvaluatePerImage(net, data, name)
  net:evaluate()
  local loss_sum = 0
  local total_num_labels = 0
  local total_num_correct = 0
  local x, yt, class_weights, filename = data:GetNextBatch()
  local num_batches = 0
  local num_classes = param_num_classes
  local softmax = cudnn.SpatialSoftMax():cuda()
  local iou_acc = torch.FloatTensor(data:size()):fill(0)
  local img_idx = 0
  local filenames = {}
  param_no_print = true
  local out_dir = '/mnt/ikreso/datasets/Cityscapes/pyramid/1504x672_8s/results/' .. name .. '/'
  --local out_dir = '/mnt/ikreso/datasets/Cityscapes/results/baseline/' .. name .. '/'
  while x do
    --local out_dir = '/mnt/ikreso/datasets/Cityscapes/softmax_potentials/' .. name .. '/'
    --os.execute('mkdir -p ' .. out_dir)
    --local img_out_dir = '/mnt/ikreso/datasets/Cityscapes/pyramid/2048x1024_8/results/' .. name .. '/'
    local img_out_dir = out_dir .. 'img/'
    os.execute('mkdir -p ' .. img_out_dir)
    img_idx = img_idx + 1
    local y
    if type(x) == 'table' then
      y = PyramidFreeForward(net, x)
    else
      y = net:forward(x)
    end
    local _, pred = y:max(2)
    pred = pred[1][1]:int()
    local pred_rgb = DrawPrediction(pred)
    image.save(img_out_dir .. filename, pred_rgb)
    filename_tensor = filename:sub(1,-5) .. '.t7'
    torch.save(img_out_dir .. filename_tensor, pred)
    torch.save(img_out_dir .. filename:sub(1,-5) .. '_gt.t7', yt)

    local scores = net:get(17).output
    scores = softmax:forward(scores)[1]
    --print(scores:size())
    --print(scores[{{},1,{}}]:sum())
    --WriteTensor(scores:float():contiguous(), out_dir .. filename)

    --state.loss.nll.weights:copy(class_weights)
    --local E = state.loss:forward(y, yt)
    --loss_sum = loss_sum + E

    --SaveResult(pred, filename)

    if name ~= 'test' then
      local confusion_matrix = torch.IntTensor(num_classes, num_classes):fill(0)
      AggregateStats(pred, yt[1]:int(), confusion_matrix)
      local pixel_acc, iou = PrintEvaluationStats(confusion_matrix, name, false)
      iou_acc[img_idx] = iou
      table.insert(filenames, filename)
      print(filename)
      print(iou)
    end
    --AggregateStatsWithDepth(pred:int(), yt[1]:int(), confusion_matrix,
    --                        depth_img, disp_correct_cnt, disp_total_cnt)
    --AggregateConfusionMatrix(pred:int(), yt[1]:int(), confusion_matrix,
    --                         depth_img, disp_correct_cnt, disp_total_cnt)
    --AddToDepthStats(error_mat, depth_img, disp_errors, disp_cnt)
    xlua.progress(img_idx, data:size())
    x, yt, class_weights, filename = data:GetNextBatch()
    collectgarbage()
  end
  torch.save(out_dir .. 'baseline_val_iou_per_image.t7', {iou_acc, filenames})
  --local loss = loss_sum / total_size
  --xlua.print('avg loss = ' .. loss .. '\n')
  --return loss, avg_pixel_acc, avg_class_acc, avg_class_acc_fn, avg_class_precision
end

function Evaluate(net, data, name)
  net:evaluate()
  local loss_sum = 0
  local total_num_labels = 0
  local total_num_correct = 0
  local x, yt, class_weights, filename = data:GetNextBatch()
  local num_batches = 0
  local num_classes = param_num_classes
  local confusion_matrix = torch.IntTensor(num_classes, num_classes):fill(0)
  local softmax = cudnn.SpatialSoftMax():cuda()
  while x do
    local out_dir = '/home/kivan/datasets/Cityscapes/pyramid/results/softmax_potentials/' .. name .. '/'
    local img_out_dir = '/home/kivan/datasets/Cityscapes/pyramid/results/rgb/' .. name .. '/'
    os.execute('mkdir -p ' .. out_dir)
    os.execute('mkdir -p ' .. img_out_dir)
    num_batches = num_batches + 1
    --local y = net:forward(x)
    local y = PyramidFreeForward(net, x)
    local _, pred = y:max(2)
    pred = pred[1][1]:int()
    --local pred_rgb = DrawPrediction(pred)
    --image.save(img_out_dir .. filename, pred_rgb)

    --for i = 16, 8, -1 do
    --  local concat = net:get(i).output
    --  print(concat:size())
    --end
    --local concat = net:get(10).output
    --local scores = net:get(17).output
    ----local scores = net:get(17).output
    --scores = softmax:forward(scores)[1]
    --print(scores:size())
    --print(scores[{{},1,{}}]:sum())
    --local score_filename = filename:sub(1,-5) .. '.bin'
    --local concat_filename = filename:sub(1,-5) .. '_concat.bin'
    --WriteTensor(scores:float():contiguous(), out_dir .. score_filename)
    --WriteTensor(concat:float():contiguous(), out_dir .. concat_filename)

    --state.loss.nll.weights:copy(class_weights)
    --local E = state.loss:forward(y, yt)
    --loss_sum = loss_sum + E

    --SaveResult(pred, filename)

    if name ~= 'test' then
      AggregateStats(pred, yt[1]:int(), confusion_matrix)
    end
    --AggregateStatsWithDepth(pred:int(), yt[1]:int(), confusion_matrix,
    --                        depth_img, disp_correct_cnt, disp_total_cnt)
    --AggregateConfusionMatrix(pred:int(), yt[1]:int(), confusion_matrix,
    --                         depth_img, disp_correct_cnt, disp_total_cnt)
    --AddToDepthStats(error_mat, depth_img, disp_errors, disp_cnt)
    xlua.progress(num_batches, data:size())
    x, yt, class_weights, filename = data:GetNextBatch()
    collectgarbage()
    if name ~= 'test' then
      if num_batches % 50 == 0 then
        PrintEvaluationStats(confusion_matrix, name)
      end
    end
  end
  if name ~= 'test' then
    PrintEvaluationStats(confusion_matrix, name)
  end
  --local loss = loss_sum / total_size
  --xlua.print('avg loss = ' .. loss .. '\n')
  --return loss, avg_pixel_acc, avg_class_acc, avg_class_acc_fn, avg_class_precision
end

param_home_dir = '/home/ikreso/'
--param_num_train_files = 60
param_num_train_files = 3
param_num_valid_files = 10
--param_num_valid_files = 5
param_iters_between_val = 5000
param_skip_train_data = true
--param_skip_val_data = true

local net_dir = '/home/kivan/source/results/semseg/torch/deploy/000_TueMar22_12:27:28/'
--local net_dir = '/home/ikreso/source/results/semseg/torch/deploy/cityscapes_baseline_MonApr4_00:54:44/'
local model_path = net_dir .. '/model_copy.lua'
_, loss, train_container, validation_container = paths.dofile(model_path)
net = torch.load(net_dir .. "net.bin")
net:evaluate()
--net:clearState()
--torch.save('test_net.bin', net)

--EvaluatePerImage(net, validation_container, 'val')

Evaluate(net, validation_container, 'val')
--Evaluate(net, train_container, 'train')

--paths.dofile('data_container_multifile_threaded_test.lua')
--test_container = DataContainerMultiFileThreadedTest {
--  data_dir = param_data_dir,
--  prefix = 'test',
--  num_files = 31
--}
--Evaluate(net, test_container, 'test')
