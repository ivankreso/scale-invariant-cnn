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


--param_data_dir = '/home/kivan/datasets/Cityscapes/1632x736/torch/'
--param_data_dir = '/mnt/ikreso/datasets/Cityscapes/1952x864/torch/'
--param_data_dir = '/home/kivan/datasets/Cityscapes/1024x448/torch/'

param_num_train_files = 10
param_num_valid_files = 10
--param_data_dir = '/home/kivan/datasets/Cityscapes/1504x672/torch/'
param_data_dir = '/home/kivan/datasets/Cityscapes/pyramid/1504x672_combined_new/single_scale/'
local train_container, validation_container = paths.dofile(include_dir .. 'load_cityscapes.lua')

--param_data_dir = '/home/kivan/datasets/KITTI/semantic_segmentation/torch/1248x384/'
--local train_container, validation_container = paths.dofile(include_dir .. 'load_kitti.lua')

--local init_vgg = false
local init_vgg = true

local conv1_sz = 64
local conv2_sz = 128
local conv3_sz = 256
local conv4_sz = 512
local conv5_sz = 512

--local fc_sz = 4096
local fc_sz = 1024
--local fc_sz = 512 speed same as 1024
local num_channel = 3
local num_output = param_num_classes
local k = 3
local p = (k - 1) / 2
local s = 1
-- 2 times slower with k2 == 3
--local k2 = 1
--local k2 = 5
--local k2 = 3
local k2 = 3
local p2 = (k2 - 1) / 2

local pool_k = 2
local pool_s = 1
local pool_p = (pool_k - 1) / 2

local function AddBatchNormalization(input_size, layer)
  -- works best
  return nn.SpatialBatchNormalization(input_size, nil, nil, false)(layer)
  --return layer
  --return cudnn.SpatialBatchNormalization(input_size, nil, nil, nil)(layer)
end

local function AddBatchNormalization4(input_size, layer)
  -- better without
  return layer
  --return nn.SpatialBatchNormalization(input_size, nil, nil, false)(layer)
end

local function AddBatchNormalization5(input_size, layer)
  return nn.SpatialBatchNormalization(input_size, nil, nil, false)(layer)
  --return layer
end

local function AddBatchNormalizationFC(input_size, layer)
  return nn.SpatialBatchNormalization(input_size, nil, nil, false)(layer)
end

local function Convolve(input_sz, output_sz, k, s, p, bottom_layer)
  return cudnn.SpatialConvolution(input_sz, output_sz, k, k, s, s, p, p)(bottom_layer)
  --return nn.SpatialConvolution(input_sz, output_sz, k, k, s, s, p, p)(bottom_layer)
end

local conv1_1 = cudnn.SpatialConvolution(num_channel, conv1_sz, k, k, s, s, p, p)()
local relu1_1 = cudnn.ReLU(true)(conv1_1)
local conv1_2 = Convolve(conv1_sz, conv1_sz, k, s, p, relu1_1)
local relu1_2 = cudnn.ReLU(true)(conv1_2)
--local pool1 = cudnn.SpatialMaxPooling(2, 2)(relu1_1)
local pool1 = cudnn.SpatialMaxPooling(2, 2)(relu1_2)

local conv2_1 = Convolve(conv1_sz, conv2_sz, k, s, p, pool1)
local relu2_1 = cudnn.ReLU(true)(conv2_1)
local conv2_2 = Convolve(conv2_sz, conv2_sz, k, s, p, relu2_1)
local relu2_2 = cudnn.ReLU(true)(conv2_2)
--local pool2 = cudnn.SpatialMaxPooling(2, 2)(relu2_1)
local pool2 = cudnn.SpatialMaxPooling(2, 2)(relu2_2)

local conv3_1 = Convolve(conv2_sz, conv3_sz, k, s, p, pool2)
--local relu3_1 = cudnn.ReLU(true)(AddBatchNormalization(conv3_sz, conv3_1))
local relu3_1 = cudnn.ReLU(true)(conv3_1)
local conv3_2 = Convolve(conv3_sz, conv3_sz, k, s, p, relu3_1)
--local relu3_2 = cudnn.ReLU(true)(AddBatchNormalization(conv3_sz, conv3_2))
local relu3_2 = cudnn.ReLU(true)(conv3_2)
--local pool3 = cudnn.SpatialMaxPooling(2, 2)(relu3_2)
local conv3_3 = Convolve(conv3_sz, conv3_sz, k, s, p, relu3_2)
--local relu3_3 = cudnn.ReLU(true)(AddBatchNormalization(conv3_sz, conv3_3))
local relu3_3 = cudnn.ReLU(true)(conv3_3)
local pool3 = cudnn.SpatialMaxPooling(2, 2)(relu3_3)

local conv4_1 = Convolve(conv3_sz, conv4_sz, k, s, p, pool3)
local relu4_1 = cudnn.ReLU(true)(AddBatchNormalization4(conv4_sz, conv4_1))
local conv4_2 = Convolve(conv4_sz, conv4_sz, k, s, p, relu4_1)
local relu4_2 = cudnn.ReLU(true)(AddBatchNormalization4(conv4_sz, conv4_2))
--local pool4 = cudnn.SpatialMaxPooling(2, 2)(relu4_2)
local conv4_3 = Convolve(conv4_sz, conv4_sz, k, s, p, relu4_2)
local relu4_3 = cudnn.ReLU(true)(AddBatchNormalization4(conv4_sz, conv4_3))
local pool4 = cudnn.SpatialMaxPooling(2, 2)(relu4_3)

local conv5_1 = Convolve(conv4_sz, conv5_sz, k, s, p, pool4)
local relu5_1 = cudnn.ReLU(true)(AddBatchNormalization5(conv5_sz, conv5_1))
local conv5_2 = Convolve(conv5_sz, conv5_sz, k, s, p, relu5_1)
local relu5_2 = cudnn.ReLU(true)(AddBatchNormalization5(conv5_sz, conv5_2))
local conv5_3 = Convolve(conv5_sz, conv5_sz, k, s, p, relu5_2)
local relu5_3 = cudnn.ReLU(true)(AddBatchNormalization5(conv5_sz, conv5_3))
--local pool5 = cudnn.SpatialMaxPooling(2, 2)(relu5_3)
-- better then (2,2)
--local pool5 = cudnn.SpatialMaxPooling(pool_k, pool_k, pool_s, pool_s, pool_p, pool_p)(relu5_3)
-- +2x512fc 84.206948091821
--local pool5 = relu5_3

--local upsample5 = relu5_2
--local upsample5 = relu5_3
--local concat45 = nn.JoinTable(2)({pool4, upsample5})
--local upsample45 = nn.SpatialUpSamplingNearest(2)(concat45)
--local concat_final = nn.JoinTable(2)({upsample45, pool3})
--local concat_size = conv5_sz + conv4_sz + conv3_sz
--local upsample5 = nn.SpatialUpSamplingNearest(2)(relu5_3)
--local concat_final = nn.JoinTable(2)({upsample5, pool3})

--local upsample5 = nn.SpatialUpSamplingNearest(2)(conv5_3)
--local concat_45 = nn.JoinTable(2)({upsample5, conv4_3})
--local upsample_45 = nn.SpatialUpSamplingNearest(2)(concat_45)
--local concat_final = nn.JoinTable(2)({upsample_45, conv3_3})
--local concat_size = conv5_sz + conv4_sz + conv3_sz

--local concat_final = relu5_3
--local concat_size = conv5_sz
--local upsample_factor = 4
local upsample_factor = 8
--local upsample_factor = 16

local upsample5 = nn.SpatialUpSamplingNearest(2)(relu5_3)
--local concat_45 = nn.JoinTable(2)({upsample5, relu4_3})
--local concat_final = nn.JoinTable(2)({upsample5, relu4_3})
--local concat_size = conv5_sz + conv4_sz
local concat_final = nn.JoinTable(2)({upsample5, pool3})
local concat_size = conv5_sz + conv3_sz

--local concat_final = relu5_3
--local concat_size = conv5_sz

--local upsample_45 = nn.SpatialUpSamplingNearest(2)(concat_45)
--local concat_final = nn.JoinTable(2)({upsample_45, relu3_3})
--local concat_size = conv5_sz + conv4_sz + conv3_sz

--local conv5_score = cudnn.SpatialConvolution(conv5_sz, num_output, 1, 1)(pool5)
--local score32 = AddBatchNormalization(num_output, conv5_3))

--local fc6 = nn.Dropout(0.5)(cudnn.ReLU(true)(cudnn.SpatialConvolution(conv5_sz, fc_sz, k2, k2, s, s, p2, p2)(pool5)))
--local fc7 = nn.Dropout(0.5)(cudnn.ReLU(true)(cudnn.SpatialConvolution(fc_sz, fc_sz, 1, 1)(fc6)))
--local score32 = cudnn.SpatialConvolution(fc_sz, num_output, 1, 1)(fc7)

--local score32 = cudnn.SpatialConvolution(conv5_sz, num_output, 1, 1)(pool5)

--local fc6 = nn.Dropout(0.5)(cudnn.ReLU(true)(cudnn.SpatialConvolution(conv5_sz, 512, 1, 1)(pool5)))
--local score32 = cudnn.SpatialConvolution(512, num_output, 1, 1)(fc6)

local fc6_conv = Convolve(concat_size, fc_sz, k2, s, p2, concat_final)
--local fc6_conv = cudnn.SpatialConvolution(concat_size, fc_sz, 1, 1)(concat_final)
--local fc6_conv = cudnn.SpatialConvolution(conv5_sz + conv3_sz, fc_sz, 1, 1)(concat_final)
--local fc6_conv = cudnn.SpatialConvolution(conv5_sz, fc_sz, 1, 1)(pool5)
local fc6 = cudnn.ReLU(true)(AddBatchNormalizationFC(fc_sz, fc6_conv))
local fc7_conv = cudnn.SpatialConvolution(fc_sz, fc_sz, 1, 1)(fc6)
local fc7 = cudnn.ReLU(true)(AddBatchNormalizationFC(fc_sz, fc7_conv))

local score32 = cudnn.SpatialConvolution(fc_sz, num_output, 1, 1)(fc7)
--local score32 = cudnn.SpatialConvolution(fc_sz, num_output, 1, 1)(fc6)
--local fc7 = cudnn.SpatialConvolution(512, num_output, 1, 1)(fc6)
--local score32 = AddBatchNormalization(num_output, fc7)
--local score32 = cudnn.SpatialConvolution(512, num_output, 1, 1)(fc6)

local final_score = nn.SpatialUpSamplingNearest(upsample_factor)(score32)

local net = nn.gModule({conv1_1}, {final_score})
if param_training then
  for i = 1, net:size() do
    print(i .. " --> ", net:get(i))
  end
end

if init_vgg and param_training then
  require 'loadcaffe'
  local vgg_model = loadcaffe.load(param_vgg_prototxt, param_vgg_model, 'cudnn')
  --InitNetwork(net, vgg_model)
  --local conv_layers = GetConvolutionLayers(net)
  --local conv_layers = {2, 4, 7, 9}
  --local conv_layers = {2, 4, 7, 9, 12, 15, 18}
  --local conv_layers = {2, 4, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37, 40}
  --local conv_layers = {2, 4, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37, 40}

  --local conv_layers = {1, 4, 8, 11, 15, 18, 21, 25, 28, 31}
  --local conv_layers = {1, 4, 8, 11, 15, 18, 21, 25, 28, 31}
  --local conv_layers = {2, 5, 9, 12, 16, 19, 22, 26, 29, 32, 36, 39, 42}
  --local conv_layers = {2, 4, 7, 10, 14, 17, 20, 24, 27, 30}
  local vgg_conv_layers = GetConvolutionLayers(vgg_model)
  --local init_idx = {1, 3, 5, 6, 8, 9, 11, 12}
  --local init_idx = {1, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13}
  --local init_idx = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}
  local j = 0
  --for i = 1, #init_idx do
  --for i = 1, #vgg_conv_layers do
  for i = 1, 13 do
  --for i = 1, 13 do
    local layer
    repeat 
      j = j + 1
      layer = net:get(j)
    until layer.weight ~= nil and layer.kW == 3

    --print("Initalizing layer: ", conv_layers[i])
    print("Initalizing layer: ", j)
    --local layer = net:get(conv_layers[i])
    layer.weight:copy(vgg_conv_layers[i].weight)
    layer.bias:copy(vgg_conv_layers[i].bias)
    --local idx = init_idx[i]
    --layer.weight:copy(vgg_conv_layers[idx].weight)
    --layer.bias:copy(vgg_conv_layers[idx].bias)
  end
  --for i = 1, #conv_layers do
  --  print("Initalizing layer: ", conv_layers[i])
  --  local layer = net:get(conv_layers[i])
  --  layer.weight:copy(vgg_conv_layers[i].weight)
  --  layer.bias:copy(vgg_conv_layers[i].bias)
  --end
  vgg_model = nil
end

collectgarbage()

--net = nn.gModule({input}, {conv1_2})
--graph.dot(net.fg, 'net graph', 'graph')
--weights[num_output] = 0.0
--local loss = cudnn.SpatialCrossEntropyCriterion(weights)
--local loss = nn.SpatialCrossEntropyCriterionWithIgnore()
local weights = torch.CudaTensor(num_output):fill(1.0)
local loss = nn.SpatialCrossEntropyCriterionWithIgnore(weights)
return net:cuda(), loss:cuda(), train_container, validation_container
--return net, loss, train_container, validation_container
