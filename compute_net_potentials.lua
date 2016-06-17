require 'cudnn'
require 'eval_helper'

param_iters_between_val = 5000

function ComputeOutputs(net, data, name)
  local out_dir = param_data_dir .. '/softmax_potentials/' .. name .. '/'
  os.execute('mkdir -p ' .. out_dir)
  net:evaluate()
  local x, yt, weights, filename = data:GetNextBatch()
  local num_batches = 0
  local softmax = cudnn.SpatialSoftMax():cuda()

  while x do
    num_batches = num_batches + 1
    net:forward(x)
    --print(net:size())
    -- net_size - 1
    local y = net:get(17).output
    --local y = net:forward(x)[1]
    --print(y:size())
    y = softmax:forward(y)[1]
    --print(y:size())
    filename = filename:sub(1,-5) .. '.bin'
    WriteTensor(y:float():contiguous(), out_dir .. filename)

    --print(y[{{},200,200}])
    xlua.progress(num_batches, data:size())
    x, yt, weights, filename = data:GetNextBatch()
    collectgarbage()
  end
end

local net_dir = '/home/ikreso/source/results/semseg/torch/results/TueMar22_12:27:28/'
--local net_dir = '/home/kivan/source/deep-learning/semantic_segmentation/output/nets/16s_MonFeb2912:11:482016/'
local model_path = net_dir .. '/model_copy.lua'
_, loss, train_container, validation_container = paths.dofile(model_path)
net = torch.load(net_dir .. "net.bin")

ComputeOutputs(net, validation_container, 'val')
ComputeOutputs(net, train_container, 'train')
