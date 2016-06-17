require('torch')
require('nn')
require('cunn')
require('cudnn')


-- common obj name to be freed
--local common = {'output', 'gradInput'}
local common = {'output', 'gradInput', 'gradWeight', 'gradBias'}

-- temporary buffer name other than output/gradInput
local t = {
  -- convolution
  ['nn.SpatialConvolution'] = {'finput', 'fgradInput'},
  ['nn.SpatialConvolutionMM'] = {'finput', 'fgradInput'},
  ['cudnn.SpatialConvolution'] = {'finput', 'fgradInput'},

  -- pooling
  ['nn.SpatialMaxPooling'] = {'indices'},
  ['nn.TemporalMaxPooling'] = {'indices'},
  ['nn.VolumetricMaxPooling'] = {'indices'},
  ['nn.SpatialFractionalMaxPooling'] = {'indices'},

  -- regularizer
  ['nn.BatchNormalization'] = {'buffer', 'buffer2', 'centered', 'normalized'},
  --['nn.SpatialBatchNormalization'] = {'running_mean', 'running_var', 'save_mean', 'save_std'},
  ['nn.Dropout'] = {'noise'},
  ['nn.SpatialDropout'] = {'noise'},

  -- transfer
  ['nn.PReLU'] = {'gradWeightBuf', 'gradWeightBuf2'},
  ['nn.LogSigmoid'] = {'buffer'},

  -- etc
  ['nn.Mean'] = {'_gradInput'},
  ['nn.Normalize'] = {'_output', 'norm', 'normp'},
  ['nn.PairwiseDistance'] = {'diff'},
  ['nn.Reshape'] = {'_input', '_gradOutput'},

  -- fbcunn
  ['nn.AbstractParallel'] = {'homeGradBuffers', 'input_gpu', 'gradOutput_gpu', 'gradInput_gpu'},
  ['nn.DataParallel'] = {'homeGradBuffers', 'input_gpu', 'gradOutput_gpu', 'gradInput_gpu'},
  ['nn.ModelParallel'] = {'homeGradBuffers', 'input_gpu', 'gradOutput_gpu', 'gradInput_gpu'},
}


local function free_table_or_tensor(val, name, field)
  if type(val[name]) == 'table' then
    val[name] = {}
  elseif type(val[name]) == 'userdata' then
    val[name] = field.new()
    --val[name] = nil
  end
end


local function is_member(name, t)
  if t == nil then
    return false
  end

  for _, value in pairs(t) do
    if name == value then
      --print('  CLEANED: ' .. name)
      return true
    end
  end
  return false
end


-- Taken and modified from Soumith's imagenet-multiGPU.torch code
-- https://github.com/soumith/imagenet-multiGPU.torch/blob/master/train.lua
local function sanitize(model)
  local list = model:listModules()
  for _, val in ipairs(list) do
    local module_name = val.__typename
    print('\n' .. module_name)
    for name, field in pairs(val) do
      --print(name .. ' -- ' .. torch.type(field))
      -- remove ffi obj
      if torch.type(field) == 'cdata' then
        --print('  CLEANED: ' .. name)
        val[name] = nil

        -- remove common obj
      elseif is_member(name, common) then
        free_table_or_tensor(val, name, field)

        -- remove specific obj
      elseif is_member(name, t[val.__typename]) then
        free_table_or_tensor(val, name, field)
      else
        print(name .. ' -- ' .. torch.type(field))
        if torch.type(field) == 'torch.FloatTensor' or torch.type(field) == 'torch.CudaTensor' then
          --print(name)
          print(field:size())
        end
      end
    end
    if module_name == "cudnn.SpatialConvolution" then
      val:clearDesc()
    end
  end
  collectgarbage()
  return model
end


return sanitize
