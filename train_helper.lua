local function GetConvolutionLayers(net)
  local layers = {}
  for i = 1, net:size() do
    if net:get(i).weight ~= nil then
      table.insert(layers, net:get(i))
    end
  end
  return layers
end

function InitWithVGG(net, vgg_model, num_layers)
  local vgg_conv_layers = GetConvolutionLayers(vgg_model)
  --local num_layers = 13
  --local num_layers = 10
  print(string.format("Init first %d VGG layers...", num_layers))
  local j = 0
  for i = 1, num_layers do
    local layer
    repeat 
      j = j + 1
      layer = net:get(j)
    until layer.weight ~= nil and layer.kW == 3

    --print("Initalizing layer: ", conv_layers[i])
    print("Initalizing layer: ", j)
    layer.weight:copy(vgg_conv_layers[i].weight)
    layer.bias:copy(vgg_conv_layers[i].bias)
  end
end

function InitDepthNetWithVGG(net, vgg_model)
  local vgg_conv_layers = GetConvolutionLayers(vgg_model)
  --local num_layers = 13
  local num_layers = 10
  print(string.format("Init first %d VGG layers...", num_layers))
  local j = 0
  for i = 1, num_layers do
    local layer
    repeat 
      j = j + 1
      layer = net:get(j)
    until layer.weight ~= nil and layer.kW == 3

    --print("Initalizing layer: ", conv_layers[i])
    print("Initalizing layer: ", j)
    if i == 1 then
      layer.weight:copy(vgg_conv_layers[i].weight[{{},{2},{},{}}])
    else
      layer.weight:copy(vgg_conv_layers[i].weight)
    end
    layer.bias:copy(vgg_conv_layers[i].bias)
  end
end


local function DrawFilters()
  if draw_filters then
    --local weights = net:get(1):get(1):get(1).weight:clone()
    --local weights = net:get(1):get(1).weight:clone()
    local filters = optim_state.net:get(2).weight:clone():float()
    --win = image.display(weights,5,nil,nil,nil,win)
    --image.saveJPG(paths.concat(save_dir, 'Filters_epoch'.. epoch .. '.jpg'), image.toDisplayTensor(weights))
    --img = gm.Image(weights:float(), 'I', 'DHW')
    --disp = image.toDisplayTensor(filters)
    -- disp is CudaTensor ????
    --image.save(save_dir .. "/img_filters.pgm", disp:float())
    --saturate=saturate, scaleeach=scaleeach, min=min, max=max, symmetric=symm}
    --print(filters:size())

    -- use this to draw filters in each dimension
    --for i = 1, filters:size()[2] do
    for i = 1, 1 do
      local disp = image.toDisplayTensor{input=filters[{{},i,{},{}}], padding=1, nrow=8}
      --print(disp:size())
      local img = gm.Image(disp:resize(torch.LongStorage({1,disp:size(1),disp:size(2)})):float(), 'I', 'DHW')
      img:save(filters_save_dir .. '/epoch_'..epoch..'_layer_' .. 1 .. '_filter_' .. i .. '.pgm')
    end
    -- use this to draw one 3D filter across all dims
    --for i = 1, filters:size()[1] do
    --  local disp = image.toDisplayTensor{input=filters[{i,{},{},{}}], padding=1, nrow=8}
    --  --print(disp:size())
    --  local img = gm.Image(disp:resize(torch.LongStorage({1,disp:size(1),disp:size(2)})):float(), 'I', 'DHW')
    --  img:save(save_dir .. '/epoch_'..epoch..'_layer_'.. 1 .. '_dim_' .. i .. '.pgm')
    --end

    --print(disp)
    --for i = 1,filters:size(1) do
    --  print(filters[i])
    --  --img = gm.Image(filters[i], 'I', 'DHW')
    --  --img = gm.Image(filters[i], 'I', 'DHW')
    --  img = gm.Image(disp[i], 'I', 'DHW')
    --  --img:save(save_dir .. '/filters_epoch'.. epoch .. '_num_' .. i .. '.jpg')
    --  img:save(save_dir .. '/filters_epoch'.. epoch .. '_num_' .. i .. '.pgm')
    --end
  end
end
