require 'image';
paths.dofile('eval_helper.lua');

local net_dir = '/home/ikreso/source/deep-learning/semantic_segmentation/output/results/TueMar15_13:31:21/'
local out_dir = net_dir .. '/predictions/'
os.execute('mkdir -p ' .. out_dir)
--local net_dir = '/home/kivan/source/deep-learning/semantic_segmentation/output/nets/wgt1000_MonMar711:18:232016/'
local model_path = net_dir .. '/model_copy.lua'
_, loss, train_container, validation_container = paths.dofile(model_path)
net = torch.load(net_dir .. "net.bin")
net:evaluate()
print(validation_container:size())

function ComputePrediction(net, data, name)
  --local out_dir = param_data_dir .. '/img/' .. name .. '/softmax_potentials/'
  --os.execute('mkdir -p ' .. out_dir)
  net:evaluate()
  local x, yt, weights, filename = data:GetNextBatch()
  local num_batches = 0
  while x do
    print(filename)
    num_batches = num_batches + 1
    local y = net:forward(x)
    local _, pred = y:max(2)
    pred = pred[1][1]:int()
    --local rgb_img = image.load(param_data_dir .. '/../img/data/' .. name .. '/' .. filename:sub(1,filename:find('_')-1) .. '/' .. filename)
    --local rgb_label = image.load(param_data_dir .. '/../img/labels/' .. name .. '/' .. filename:sub(1,filename:find('_')-1) .. '/' .. filename)
    local pred_rgb = DrawPrediction(pred)
    --yt = yt[1]:int()
    --local mask = yt:eq(pred)
    --mask[yt:eq(0)] = 1
    --pred_rgb[1][mask] = 0
    --pred_rgb[2][mask] = 0
    --pred_rgb[3][mask] = 0
    --pred_rgb[{{},mask}] = 0
    --itorch.image(pred_rgb)
    image.save(out_dir .. filename, pred_rgb)

    x, yt, weights, filename = data:GetNextBatch()
    collectgarbage()
  end
end

ComputePrediction(net, validation_container, 'val')
