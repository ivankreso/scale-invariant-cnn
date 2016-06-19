--require('mobdebug').start()

require 'torch'
require 'cutorch'
require 'cunn'
require 'optim'
require 'xlua'
require 'image'
paths.dofile('eval_helper.lua')

function IsNaN(v)
  return v ~= v
end


local function PrintGradientStats(layers)
  print('')
  for i = 1, layers:size() do
    --if state.net:get(i).gradWeight ~= nil then
    local layer = layers:get(i)
    if layer.gradInput ~= nil then
      --local gradInput = layer.gradInput:float()
      --local size = gradInput:size()
      --gradInput:abs()
      --gradInput = gradInput:view(-1)
      --local name = "layer"..i.." size = "..size[1]..'x'..size[2]..'x'..size[3]..'x'..size[4]
      --print(name .. ' gradInput: mean = ' .. gradInput:mean() .. '  median = ' ..
      --      gradInput:median()[1])
      if layer.gradWeight ~= nil and layer.kW ~= nil then
        local gradWeight = layer.gradWeight:float()
        local size = gradWeight:size()
        gradWeight:abs()
        gradWeight = gradWeight:view(-1)
        local name = "layer"..i.." size = "..size[1]..'x'..size[2]..'x'..size[3]..'x'..size[4]
        print(name .. ' gradWeight: mean = ' .. gradWeight:mean() .. '  median = ' ..
          gradWeight:median()[1])
      end
      --plot = Plot():histogram(state.net:get(i).gradWeight:view(-1):float()):draw()
      --plot:title():redraw()
    end
  end
end


local function PrintTrainingStats(init_net, net)
  print('Param dist from init in each layer')
  for i = 1, net:size() do
    local layer = net:get(i)
    local init_layer = init_net:get(i)
    print(layer)
    if layer.weight ~= nil then
      local size = layer.weight:size()
      local dist = (layer.weight:float() - init_layer.weight):abs():sum() / init_layer.weight:nElement()
      print('weight dist = ', dist)
    end
    if layer.bias ~= nil then
      local dist = (layer.bias:float() - init_layer.bias):abs():sum() / init_layer.bias:nElement()
      print('bias dist = ', dist)
    end
  end
end


local function PrintOptimizerState(state)
  local lr = state.learningRate or 1e-3
  local lrd = state.learningRateDecay or 0
  local nevals = state.evalCounter
  if lr ~= nil and lrd ~= nil and nevals ~= nil then
    print("----Optimizer State----")
    print("LR = " .. lr / (1 + nevals*lrd))
  end
end


local function Optimize(state, input, yt, class_weights)
  -- define eval closure
  local err = 0
  local y
  local feval = function(params)
    -- get new parameters
    if params ~= state.weights then
      state.weights:copy(params)
    end
    -- reset gradients
    state.gradients:zero()
    -- feed it to the neural network and the criterion
    y = state.net:forward(input)
    state.loss.nll.weights:copy(class_weights)
    err = state.loss:forward(y, yt)

    local loss_grad = state.loss:backward(y, yt)
    state.net:backward(input, loss_grad)
    return err, state.gradients
  end
  state.optim_function(feval, state.weights, state.optim_conf)
  return err, y
end

function Train(state) 
  state.net:training()
  local iter = 0
  local data = state.train_container
  local x, labels, class_weights, filename = data:GetNextBatch()
  local loss_sum = 0
  local num_labels = 0
  local num_classes = param_num_classes
  local confusion_matrix = torch.IntTensor(num_classes, num_classes):fill(0)
  local print_stride = torch.round(data:size() / 5)
  if param_dataset_name == 'cityscapes' then
    print_stride = torch.round(data:size() / 30)
  end
  while x do
    --state.num_batches = state.num_batches + 1
    local E, y = Optimize(state, x, labels, class_weights)
    --loss_sum = loss_sum + (E / labels:ne(0):sum())
    num_labels = num_labels + labels:ne(0):sum()
    loss_sum = loss_sum + E

    local _, pred = y:max(2)
    pred = pred[1][1]
    AggregateStats(pred:int(), labels[1]:int(), confusion_matrix)

    x, labels, class_weights, filename = data:GetNextBatch()
    collectgarbage()

    if iter > 0 and iter % print_stride == 0 then
      local avg_loss = loss_sum / num_labels
      xlua.print('\nIter ' .. iter .. ': avg loss = ', avg_loss)
      if plot_graphs then
        if state.loss_plot:size():size() == 0 then
          state.loss_plot = torch.FloatTensor({avg_loss})
        else
          state.loss_plot = torch.cat(state.loss_plot, torch.FloatTensor({avg_loss}))
        end
      end
    end
    if opt.show_gradients then
      PrintGradientStats(state.net)
    end
    iter = iter + 1
    data:PrintProgress()
  end
  PrintEvaluationStats(confusion_matrix, 'Train')
  return loss_sum / num_labels
end

function Validate(state)
  state.net:evaluate()
  local loss_sum = 0
  local total_num_labels = 0
  local total_num_correct = 0
  local data = state.validation_container
  local x, yt, class_weights, filename = data:GetNextBatch()
  local num_batches = 0
  local num_classes = param_num_classes
  local confusion_matrix = torch.IntTensor(num_classes, num_classes):fill(0)
  local max_disp = 200
  --local disp_correct_cnt = torch.FloatTensor(num_classes, max_disp):fill(0)
  --local disp_total_cnt = torch.FloatTensor(num_classes, max_disp):fill(0)
  while x do
    num_batches = num_batches + 1
    local y = state.net:forward(x)
    --local dist = y:clone():float()
    --print(nn.SoftMax():forward(dist):sum(2))
    --print(nn.SoftMax():forward(dist):max())

    state.loss.nll.weights:copy(class_weights)
    local E = state.loss:forward(y, yt)
    --local E = state.loss:forward(yt:view(1, yt:size(1), yt:size(2), yt:size(3)), yt)
    loss_sum = loss_sum + E
    local _, pred = y:max(2)
    pred = pred[1][1]

    AggregateStats(pred:int(), yt[1]:int(), confusion_matrix)
    xlua.progress(num_batches, data:size())
    x, yt, class_weights, filename = data:GetNextBatch()
    collectgarbage()
  end
  local avg_pixel_acc, avg_class_acc, avg_class_acc_fn, avg_class_precision, total_size =
    PrintEvaluationStats(confusion_matrix, 'Validation')
  local loss = loss_sum / total_size
  xlua.print('Validation avg loss = ' .. loss .. '\n')
  return loss, avg_pixel_acc, avg_class_acc, avg_class_acc_fn, avg_class_precision
end

torch.setdefaulttensortype('torch.FloatTensor')

cmd = torch.CmdLine()
cmd:addTime()

cmd:text('Platform Optimization')
cmd:option('-threads', 1, 'number of threads')
cmd:option('-devid', 1, 'device ID (if using CUDA)')
cmd:option('-model', '', 'model path')
cmd:option('-resume', '', 'net dir to resume train')
cmd:option('-solver', '', 'solver config path')
cmd:option('-show_gradients', false, '')
cmd:option('-dont_save', false, '')

opt = cmd:parse(arg or {})
local timer = torch.Timer()

if (opt.model == '' and opt.resume == '') or opt.solver == '' then
  print('Missing config')
end

paths.dofile(opt.solver)

-- randperm and some other funcs don't work with cuda... dont ever use this
--torch.setdefaulttensortype('torch.CudaTensor')
torch.setnumthreads(opt.threads)
cutorch.setDevice(opt.devid)

local draw_filters = false
-- Globals:
local plot_graphs = false

local model_path
if opt.model ~= '' then
  local time_str = os.date():gsub(' ','')
  local dir_name
  if #time_str == 20 then
    dir_name = time_str:sub(1,8) .. '_' .. time_str:sub(9,16)
  else
    dir_name = time_str:sub(1,7) .. '_' .. time_str:sub(8,15)
  end
  save_dir = param_home_dir ..  'source/results/semseg/torch/results/' .. dir_name .. '/'
  model_path = opt.model
else
  save_dir = opt.resume .. '/'
  model_path = save_dir .. 'model.lua'
end
local filters_save_dir = save_dir .. 'filters/'
img_save_dir = save_dir .. 'imgs/'
stats_save_dir = save_dir .. 'stats/'
local model_filename = save_dir .. 'net.bin'

print('Save dir = ', save_dir)
print('Solver policy =', param_lr_policy)

----------------------------------------------------------------------
-- Model + Loss:
local net, loss, train_container, validation_container = paths.dofile(model_path)
local init_net = net:clone():float()
local weights, gradients = net:getParameters()

if opt.resume ~= '' then
  print("Resuming training using: " .. model_filename)
  net = torch.load(model_filename)
end
collectgarbage()
print('Number of network params = ', weights:size(1))
assert(gradients:size(1) == weights:size(1))

if opt.resume == '' then
  os.execute('mkdir -p ' .. save_dir)
  os.execute('mkdir -p ' .. filters_save_dir)
  os.execute('mkdir -p ' .. img_save_dir)
  os.execute('mkdir -p ' .. stats_save_dir)
  os.execute('ln -s ' .. model_path .. ' ' .. save_dir .. '/model.lua')
  os.execute('cp ' .. model_path .. ' ' .. save_dir .. '/model_copy.lua')
  os.execute('cp ' .. opt.solver .. " " .. save_dir .. '/solver_config.lua')
end


local optim_state = {
  net = net,
  loss = loss,
  weights = weights,
  clean_net_weights = clean_net_weights,
  gradients = gradients,
  optim_function = _G.optim[param_optimization],
  optim_conf = param_optim_conf,
  train_container = train_container,
  validation_container = validation_container,
  loss_plot = torch.FloatTensor()
}

error_logfile = torch.DiskFile(stats_save_dir .. 'error_log.txt', "w")
local train_loss_log = torch.FloatTensor(1):fill(0)
local valid_loss_log = torch.FloatTensor(1):fill(0)
local valid_accuracy_log = torch.FloatTensor(1):fill(0)
local valid_class_accuracy_log = torch.FloatTensor(1):fill(0)
local best_valid_loss = 1e10
local best_train_loss = 1e10
local best_valid_accuracy = 0
local best_valid_class_acc = 0
local best_valid_class_acc_fn = 0
local best_valid_precision = 0
local final_pixel_accuracy = 0
local final_recall = 0
local final_precision = 0
local best_bet = nil
local epoch = 1
print '\n--> Starting Training\n'
while epoch ~= param_max_epoch do
  if param_lr_policy[epoch] ~= nil then
    optim_state.optim_conf.learningRate = param_lr_policy[epoch]
  end
  print("Optimizer state:\n", optim_state.optim_conf)

  xlua.print('\nTraining epoch ' .. epoch)
  local train_loss = Train(optim_state)
  xlua.print('Train loss = ' .. train_loss .. '\n')
  if train_loss < best_train_loss then
    best_train_loss = train_loss
  end
  local valid_loss, valid_accuracy, valid_class_acc, valid_class_acc_fn,
    valid_precision = Validate(optim_state)

  if valid_loss < best_valid_loss then
    best_valid_loss = valid_loss
  end
  if valid_accuracy > best_valid_accuracy then
    best_valid_accuracy = valid_accuracy
  end
  if valid_class_acc_fn > best_valid_class_acc_fn then
    best_valid_class_acc_fn = valid_class_acc_fn
  end
  if valid_precision > best_valid_precision then
    best_valid_precision = valid_precision
  end
  if valid_class_acc > best_valid_class_acc then
    if not opt.dont_save then
      print('Saving network...')
      optim_state.net:clearState()
      torch.save(model_filename, optim_state.net)
    end
    best_valid_class_acc = valid_class_acc
    final_pixel_accuracy = valid_accuracy
    final_recall = valid_class_acc_fn
    final_precision = valid_precision
  end
  PrintAndSave("Best pixel accuracy = \t\t" .. best_valid_accuracy)
  PrintAndSave("Best mean class IoU accuracy TP/(TP+FP+FN) = \t" .. best_valid_class_acc)
  print("\tpixel accuracy = \t\t" .. final_pixel_accuracy)
  print("\tmean class recall - TP/(TP+FN) = \t" .. final_recall)
  print("\tmean class precision - TP/(TP+FP) = \t" .. final_precision)
  --print("Best mean class recall - TP/(TP+FN) = \t", best_valid_class_acc_fn)
  --print("Best mean class precision - TP/(TP+FP) = ", best_valid_precision)
  print('')
  --PrintTrainingStats(init_net, net)

  if epoch == 1 then
    train_loss_log[1] = train_loss
    valid_loss_log[1] = valid_loss
    valid_accuracy_log[1] = valid_accuracy
    valid_class_accuracy_log[1] = valid_class_acc
  else
    train_loss_log = torch.cat(train_loss_log, torch.FloatTensor({train_loss}))
    valid_loss_log = torch.cat(valid_loss_log, torch.FloatTensor({valid_loss}))
    valid_accuracy_log = torch.cat(valid_accuracy_log, torch.FloatTensor({valid_accuracy}))
    valid_class_accuracy_log = torch.cat(valid_class_accuracy_log, torch.FloatTensor({valid_class_acc}))
  end

  local plot_loss = {{'train loss', train_loss_log}, {'valid loss', valid_loss_log}}
  local plot_accuracy = {{'pixel accuracy', valid_accuracy_log},
                         {'class accuracy', valid_class_accuracy_log}}
  torch.save(stats_save_dir .. 'plot_loss.t7', plot_loss)
  torch.save(stats_save_dir .. 'plot_accuracy.t7', plot_accuracy)

  epoch = epoch + 1
  print('Time elapsed: ' .. timer:time().real / 60 .. ' min')
end
