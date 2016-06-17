gnuplot = require 'gnuplot'
require 'eval_helper'

function GetStats(net, data)
  net:evaluate()
  local loss_sum = 0
  local total_num_labels = 0
  local total_num_correct = 0
  local x, yt, filename, depth_img = data:GetNextBatch()
  local num_batches = 0
  local num_classes = param_num_classes
  local confusion_matrix = torch.IntTensor(num_classes, num_classes):fill(0)
  local max_disp = 200
  --local disp_correct_cnt = torch.FloatTensor(max_disp):fill(0)
  --local disp_total_cnt = torch.FloatTensor(max_disp):fill(0)
  local disp_correct_cnt = torch.FloatTensor(num_classes, max_disp):fill(0)
  local disp_total_cnt = torch.FloatTensor(num_classes, max_disp):fill(0)
  while x do
    num_batches = num_batches + 1
    local y = net:forward(x)
    --local E = loss:forward(y, yt)
    --loss_sum = loss_sum + E
    local _, pred = y:max(2)
    pred = pred[1][1]

    --SaveResult(pred, filename)
    local ignore_mask = yt:ne(0)
    --local num_labels = ignore_mask:sum()
    --total_num_labels = total_num_labels + num_labels
    local error_mat = pred:eq(yt):cmul(ignore_mask)
    --local num_correct = error_mat:sum()
    --total_num_correct = total_num_correct + num_correct

    AggregateStats(pred:int(), yt[1]:int(), confusion_matrix,
                   depth_img, disp_correct_cnt, disp_total_cnt)
    --AggregateConfusionMatrix(pred:int(), yt[1]:int(), confusion_matrix,
    --                         depth_img, disp_correct_cnt, disp_total_cnt)
    --AddToDepthStats(error_mat, depth_img, disp_errors, disp_cnt)
    --xlua.print("\nPixel error = ", 100.0 * num_errors / num_labels)
    --local loss_val = E / yt:ne(0):sum()
    --xlua.print("Loss value = ", loss_val)

    --err = err + loss_helper.ErrorCount(y, labels)
    --dist_err = dist_err + loss_helper.DistanceErrorSum(y, labels)
    --print(y)
    --xlua.print('dist_err = ', dist_err / num_examples)
    --xlua.print('percent_err = ', err / num_examples)
    xlua.progress(num_batches, data:size())
    x, yt, filename, depth_img = data:GetNextBatch()
    collectgarbage()
  end
  --xlua.print(disp_correct_cnt)
  local plot_data = {}
  local class_errors_per_disp = disp_total_cnt - disp_correct_cnt
  local pixels_per_disp = disp_total_cnt:sum(1)[1]
  local errors_per_disp = class_errors_per_disp:sum(1)[1]
  local acc_per_disp = disp_correct_cnt:sum(1)[1]

  local acc_per_class = disp_correct_cnt:clone()
  acc_per_class:cdiv(disp_total_cnt)
  for i = 1, num_classes do
    gnuplot.figure()
    gnuplot.title(param_label_colors[i][4])
    gnuplot.plot({"total", disp_total_cnt[i]}, {"errors", class_errors_per_disp[i]})
    gnuplot.figure()
    gnuplot.title(param_label_colors[i][4] .. " accuracy")
    gnuplot.plot({"accuracy", acc_per_class[i]})
  end

  gnuplot.figure()
  gnuplot.plot({"pixels per disp", pixels_per_disp}, {"errors per disp", errors_per_disp})
  --gnuplot.plot({"pixels per disp", pixels_per_disp}, {"errors per disp", errors_per_disp},
  --{"corrct", acc_per_disp})
  acc_per_disp:cdiv(pixels_per_disp)
  gnuplot.figure()
  gnuplot.plot({"accuracy per disp", acc_per_disp})
  --local disp_acc = disp_correct_cnt:cdiv(disp_total_cnt)
  ----xlua.print(disp_acc)
  --gnuplot.figure()
  --gnuplot.plot({'depth accuracy', disp_acc})

  --for i = 1, 3 do
  --  disp_correct_cnt[i]:cdiv(disp_total_cnt[i]:add(1))
  --  --gnuplot.figure(5+i)
  --  --gnuplot.plot({'class_'..i, disp_acc})
  --  table.insert(plot_data, {'class '..i, disp_correct_cnt[i]})
  --end

  xlua.print(confusion_matrix)
  --local avg_pixel_acc2 = total_num_correct / total_num_labels
  local num_correct = confusion_matrix:trace()
  local total_size = confusion_matrix:sum()
  local avg_pixel_acc = num_correct / total_size
  local size_per_class = confusion_matrix:sum(1)[1]
  local class_acc = torch.FloatTensor(size_per_class:size(1))
  xlua.print(string.format("Valid avg pixel accuracy = %.2f %%", avg_pixel_acc * 100.0))
  for i = 1, size_per_class:size(1) do
    class_acc[i] = confusion_matrix[{i,i}] / size_per_class[i] * 100.0
    xlua.print(string.format("\t%s accuracy = %.2f %%", param_label_colors[i][4], class_acc[i]))
  end
  local avg_class_acc = class_acc:mean()
  xlua.print(string.format("Valid avg class accuracy = %.2f %%", avg_class_acc))
  --return (err/DataC:size())
  --return (err/num_examples), (dist_err / num_examples)
  return loss_sum / total_size, avg_pixel_acc, avg_class_acc
end

local net_dir = '/home/kivan/source/deep-learning/semantic_segmentation/output/results/WedFeb310:46:492016/'

local model_path = net_dir .. '/model_copy.lua'
local net, loss, train_container, validation_container = paths.dofile(model_path)
net = torch.load(net_dir .. "4_epoch_net.t7")
--local weights, gradients = net:getParameters()
--BUG
--local weights_filename = net_dir .. '/net_weights.t7'
--local learned_params = torch.load(weights_filename)
--print(net:get(2).weight[1])
--weights:copy(learned_params)
--print(net:get(2).weight[1])
net:evaluate()

GetStats(net, validation_container)

local stats_dir = net_dir .. '/stats/'
local train_loss_data = torch.load(stats_dir .. 'plot_loss.t7')
gnuplot.figure()
--gnuplot.plot(train_loss_data)
gnuplot.plot(train_loss_data[1], train_loss_data[2])

local train_accuracy_data = torch.load(stats_dir .. 'plot_accuracy.t7')
gnuplot.figure()
--gnuplot.plot(train_accuracy_data)
gnuplot.plot(train_accuracy_data[1], train_accuracy_data[2])

--local depth_accuracy_data = torch.load(stats_dir .. 'plot_depth_accuracy_per_class.t7')
--gnuplot.figure()
----gnuplot.plot(depth_accuracy_data)
--gnuplot.plot(train_accuracy_data[1], train_accuracy_data[2])

--local depth_errors_data = torch.load(stats_dir .. 'plot_depth_errors.t7')
--gnuplot.figure()
--gnuplot.plot(depth_errors_data)

io.read()
gnuplot.closeall()
os.execute('pkill gnuplot')

