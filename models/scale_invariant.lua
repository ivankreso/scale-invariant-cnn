require 'nngraph'
require 'cunn'
require 'cudnn'

local function GetConvolutionLayers(net)
  local layers = {}
  for i = 1, net:size() do
    if net:get(i).weight ~= nil then
      table.insert(layers, net:get(i))
    end
  end
  return layers
end

local include_dir = param_home_dir .. '/source/deep-learning/semantic_segmentation/torch/'
paths.dofile(include_dir .. 'layers/SpatialCrossEntropyCriterionWithIgnore.lua')
paths.dofile(include_dir .. 'layers/PyramidMultiplexer.lua')
paths.dofile('../../train_helper.lua')

--param_num_train_files = 30
--param_num_valid_files = 5
--param_data_dir = '/home/kivan/datasets/Cityscapes/pyramid/1504x672_8s/'
param_data_dir = '/mnt/ikreso/datasets/Cityscapes/pyramid/1504x672_8s/'
--param_data_dir = '/home/kivan/datasets/Cityscapes/pyramid/1504x672_combined_new/'
local train_container, validation_container = paths.dofile('../../load_cityscapes.lua')

--paths.dofile(include_dir .. 'data_container_pyramid.lua')
--param_data_dir = param_data_root .. '/datasets/KITTI/semantic_segmentation/torch/pyramid_scale_8/'
--local train_container, validation_container = paths.dofile(include_dir .. 'load_kitti.lua')

--local init_vgg = false
local init_vgg = true


local conv1_sz = 64
local conv2_sz = 128
local conv3_sz = 256
local conv4_sz = 512
local conv5_sz = 512

local fc_sz = 1024
local num_channel = 3
local num_output = param_num_classes
local k = 3
local p = (k - 1) / 2
local s = 1
local k2 = 7
--local k2 = 5
--local k2 = 1
local p2 = (k2 - 1) / 2

local pool_k = 2
local pool_s = 1
local pool_p = (pool_k - 1) / 2

local function AddBatchNormalization(input_size, layer)
  --return nn.SpatialBatchNormalization(input_size, nil, nil, false)(layer)
  return layer
end

local function AddBatchNormalization4(input_size, layer)
  --return nn.SpatialBatchNormalization(input_size, nil, nil, false)(layer)
  return layer
end

local function AddBatchNormalization5(input_size, layer)
  return nn.SpatialBatchNormalization(input_size, nil, nil, false)(layer)
end

local function AddBatchNormalizationFC(input_size, layer)
  return nn.SpatialBatchNormalization(input_size, nil, nil, false)(layer)
end

local function Convolve(input_sz, output_sz, k, s, p, bottom_layer)
  return cudnn.SpatialConvolution(input_sz, output_sz, k, k, s, s, p, p)(bottom_layer)
end

local conv1_1 = cudnn.SpatialConvolution(num_channel, conv1_sz, k, k, s, s, p, p)()
local relu1_1 = cudnn.ReLU(true)(conv1_1)
local conv1_2 = Convolve(conv1_sz, conv1_sz, k, s, p, relu1_1)
local relu1_2 = cudnn.ReLU(true)(conv1_2)
local pool1 = cudnn.SpatialMaxPooling(2, 2)(relu1_2)

local conv2_1 = Convolve(conv1_sz, conv2_sz, k, s, p, pool1)
local relu2_1 = cudnn.ReLU(true)(conv2_1)
local conv2_2 = Convolve(conv2_sz, conv2_sz, k, s, p, relu2_1)
local relu2_2 = cudnn.ReLU(true)(conv2_2)
local pool2 = cudnn.SpatialMaxPooling(2, 2)(relu2_2)

local conv3_1 = Convolve(conv2_sz, conv3_sz, k, s, p, pool2)
local relu3_1 = cudnn.ReLU(true)(AddBatchNormalization(conv3_sz, conv3_1))
local conv3_2 = Convolve(conv3_sz, conv3_sz, k, s, p, relu3_1)
local relu3_2 = cudnn.ReLU(true)(AddBatchNormalization(conv3_sz, conv3_2))
local conv3_3 = Convolve(conv3_sz, conv3_sz, k, s, p, relu3_2)
local relu3_3 = cudnn.ReLU(true)(AddBatchNormalization(conv3_sz, conv3_3))
local pool3 = cudnn.SpatialMaxPooling(2, 2)(relu3_3)

local conv4_1 = Convolve(conv3_sz, conv4_sz, k, s, p, pool3)
local relu4_1 = cudnn.ReLU(true)(AddBatchNormalization4(conv4_sz, conv4_1))
local conv4_2 = Convolve(conv4_sz, conv4_sz, k, s, p, relu4_1)
local relu4_2 = cudnn.ReLU(true)(AddBatchNormalization4(conv4_sz, conv4_2))
local conv4_3 = Convolve(conv4_sz, conv4_sz, k, s, p, relu4_2)
local relu4_3 = cudnn.ReLU(true)(AddBatchNormalization4(conv4_sz, conv4_3))
local pool4 = cudnn.SpatialMaxPooling(2, 2)(relu4_3)

local conv5_1 = Convolve(conv4_sz, conv5_sz, k, s, p, pool4)
local relu5_1 = cudnn.ReLU(true)(AddBatchNormalization5(conv5_sz, conv5_1))
local conv5_2 = Convolve(conv5_sz, conv5_sz, k, s, p, relu5_1)
local relu5_2 = cudnn.ReLU(true)(AddBatchNormalization5(conv5_sz, conv5_2))
local conv5_3 = Convolve(conv5_sz, conv5_sz, k, s, p, relu5_2)
local relu5_3 = cudnn.ReLU(true)(AddBatchNormalization5(conv5_sz, conv5_3))

local upsample5 = nn.SpatialUpSamplingNearest(2)(relu5_3)
local pool5 = nn.JoinTable(2)({upsample5, pool3})
local concat_size = 3*(conv5_sz+conv3_sz)

local net = nn.gModule({conv1_1}, {pool5})
if param_training then
  for i = 1, net:size() do
    print(i .. " --> ", net:get(i))
  end
end

if init_vgg and param_training then
  require 'loadcaffe'
  local vgg_model = loadcaffe.load(param_vgg_prototxt, param_vgg_model, 'cudnn')
  InitWithVGG(net, vgg_model, 13)
  vgg_model = nil
end

collectgarbage()

local mux_input = {net()}
for i = 1, 7 do
  table.insert(mux_input, net:clone('weight', 'bias', 'gradWeight', 'gradBias')())
end
local routing_data = nn.Identity()()
table.insert(mux_input,  routing_data)
local mux = nn.PyramidMultiplexer()(mux_input)

local fc6_conv = Convolve(concat_size, fc_sz, k2, s, p2, mux)
local fc6 = cudnn.ReLU(true)(AddBatchNormalizationFC(fc_sz, fc6_conv))
local fc7_conv = cudnn.SpatialConvolution(fc_sz, fc_sz, 1, 1)(fc6)
local fc7 = cudnn.ReLU(true)(AddBatchNormalizationFC(fc_sz, fc7_conv))
local score = cudnn.SpatialConvolution(fc_sz, num_output, 1, 1)(fc7)
local upsample_factor = 8
local final_score = nn.SpatialUpSamplingNearest(upsample_factor)(score)

local final_net = nn.gModule(mux_input, {final_score})

local weights = torch.CudaTensor(num_output):fill(1.0)
local loss = nn.SpatialCrossEntropyCriterionWithIgnore(weights)
return final_net:cuda(), loss:cuda(), train_container, validation_container
