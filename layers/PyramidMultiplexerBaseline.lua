if is_defined_PyramidMultiplexer then
  return
end
is_defined_PyramidMultiplexer = true

require 'nn'
require 'nnx'

local PyramidMultiplexer, parent = torch.class('nn.PyramidMultiplexer', 'nn.Module')

--function PyramidMultiplexer:__init(owidth, oheight)
function PyramidMultiplexer:__init(width, height)
  parent.__init(self)
  self.gradInput = {}
  self.owidth = widht
  self.oheight = height
  --self.in_sz = 512
  --self.num_scales = 3
  --self.output = torch.CudaTensor(self.out_sz, self.oheight, self.owidth)
  self.output = torch.CudaTensor()
  --self.size = torch.LongStorage()

  self.upsample = nn.SpatialReSamplingEx{owidth=width, oheight=height, yDim=3, xDim=4, mode='simple'}:float()
  self.concat = nn.JoinTable(2)
end

--function PyramidMultiplexer:SetRoutingData(routing_data)
--  self.routing = routing_data
--  print('route set')
--  --print(self.routing)
--end

function PyramidMultiplexer:updateOutput(input)
  local num_inputs = #input
  self.num_scales = #input[num_inputs]
  local inputs = {input[1]:float()}
  for i = 2, #input do
    print(input[i]:size())
    local inx = input[i]:float()
    local out = self.upsample:forward(inx)
    print(out:size())
  end
  self.concat:forward(inputs)

  ----self.oheight = input[num_inputs][1]:size(1)
  ----self.owidth = input[num_inputs][1]:size(2)
  --self.in_sz = input[1]:size(2)
  --self.out_sz = self.num_scales * self.in_sz
  --self.output:resize(self.out_sz, self.oheight, self.owidth)
  ----self.out_sz = self.num_scales * self.in_sz
  ----self.output = torch.CudaTensor(self.out_sz, self.oheight, self.owidth)
  --self.routing = input[num_inputs]
  ----self.output = self.output:transpose(1, 2):transpose(2, 3):contiguous()
  ----print(self.output:size())

  --local offset = 1
  --for i = 1, self.num_scales do
  --  local routing = torch.data(self.routing[i]:contiguous())
  --  --local slice = self.output:narrow(3, offset, self.in_sz)
  --  local slice = self.output:narrow(1, offset, self.in_sz)
  --  offset = offset + self.in_sz
  --  for y = 1, self.oheight do
  --    local y_rstride = (y-1) * (self.owidth*3)
  --    for x = 1, self.owidth do
  --      local rstride = y_rstride + (3*(x-1))
  --      --print(slice:size())
  --      --local pyr_idx = routing[rstride]
  --      local pyr_idx = i
  --      local pyr_y = routing[rstride + 1]
  --      local pyr_x = routing[rstride + 2]
  --      --print(pyr_idx, pyr_x, pyr_y)
  --      slice[{{}, y, x}]:copy(input[pyr_idx][{1, {}, pyr_y, pyr_x}])
  --      --slice[{y, x}]:copy(input[pyr_idx][{1, {}, pyr_y, pyr_x}])
  --    end
  --  end
  --end
  ----self.output = self.output:transpose(2, 3):transpose(1, 2):contiguous()
  --collectgarbage()
----  self.oheight = input[1]:size(3)
----  self.owidht = input[1]:size(4)
----  nn.SpatialReSamplingEx{owidth=self.owidht, oheight=self.oheight,
----			 yDim=3, xDim=4, mode='simple'}
  ----local dimension = self:_getPositiveDimension(input)

  --return self.output:view(1, self.out_sz, self.oheight, self.owidth)
end

--function PyramidMultiplexer:updateOutput(input)
--  local num_inputs = #input
--  self.num_scales = #input[num_inputs]
--  self.oheight = input[num_inputs][1]:size(1)
--  self.owidth = input[num_inputs][1]:size(2)
--  self.in_sz = input[1]:size(2)
--  self.out_sz = self.num_scales * self.in_sz
--  self.output:resize(self.out_sz, self.oheight, self.owidth)
--  --self.out_sz = self.num_scales * self.in_sz
--  --self.output = torch.CudaTensor(self.out_sz, self.oheight, self.owidth)
--  self.routing = input[num_inputs]
--  --self.output = self.output:transpose(1, 2):transpose(2, 3):contiguous()
--  --print(self.output:size())
--
--
--  local offset = 1
--  for i = 1, self.num_scales do
--    local routing = torch.data(self.routing[i]:contiguous())
--    --local slice = self.output:narrow(3, offset, self.in_sz)
--    local slice = self.output:narrow(1, offset, self.in_sz)
--    offset = offset + self.in_sz
--    for y = 1, self.oheight do
--      local y_rstride = (y-1) * (self.owidth*3)
--      for x = 1, self.owidth do
--        local rstride = y_rstride + (3*(x-1))
--        --print(slice:size())
--        --local pyr_idx = routing[rstride]
--        local pyr_idx = i
--        local pyr_y = routing[rstride + 1]
--        local pyr_x = routing[rstride + 2]
--        --print(pyr_idx, pyr_x, pyr_y)
--        slice[{{}, y, x}]:copy(input[pyr_idx][{1, {}, pyr_y, pyr_x}])
--        --slice[{y, x}]:copy(input[pyr_idx][{1, {}, pyr_y, pyr_x}])
--      end
--    end
--  end
--  --self.output = self.output:transpose(2, 3):transpose(1, 2):contiguous()
--  collectgarbage()
----  self.oheight = input[1]:size(3)
----  self.owidht = input[1]:size(4)
----  nn.SpatialReSamplingEx{owidth=self.owidht, oheight=self.oheight,
----			 yDim=3, xDim=4, mode='simple'}
--  --local dimension = self:_getPositiveDimension(input)
--
--  return self.output:view(1, self.out_sz, self.oheight, self.owidth)
--end

function PyramidMultiplexer:updateGradInput(input, gradOutput)
  --print('back start')
  local gradInput = self.gradInput
  --print(input)
  --for i = 1, #input do
  for i = 1, #input-1 do
    if self.gradInput[i] == nil then
      self.gradInput[i] = input[i].new()
    end
    self.gradInput[i]:resizeAs(input[i]):fill(0)
    --print(self.gradInput[i]:sum())
  end
  if self.gradInput[#input] == nil then
    self.gradInput[#input] = {}
  end

  --print(gradInput)
  --print(gradOutput:size())
  local offset = 1
  for i = 1, self.num_scales do
    local routing = torch.data(self.routing[i]:contiguous())
    --local slice = self.output:narrow(3, offset, self.in_sz)
    --local slice = self.output:narrow(1, offset, self.in_sz)
    local slice = gradOutput:narrow(2, offset, self.in_sz)
    offset = offset + self.in_sz
    for y = 1, self.oheight do
      local y_rstride = (y-1) * (self.owidth*3)
      for x = 1, self.owidth do
        local rstride = y_rstride + (3*(x-1))
        --print(slice:size())
        --local pyr_idx = routing[rstride]
        local pyr_idx = i
        local pyr_y = routing[rstride + 1]
        local pyr_x = routing[rstride + 2]
        --print(pyr_idx, pyr_x, pyr_y)
        --slice[{{}, y, x}]:copy(input[pyr_idx][{1, {}, pyr_y, pyr_x}])
        gradInput[pyr_idx][{1, {}, pyr_y, pyr_x}]:add(slice[{1, {}, y, x}])
        --print(gradInput[pyr_idx][{1, {}, pyr_y, pyr_x}])
        --print(gradInput[pyr_idx]:sum())
        --slice[{y, x}]:copy(input[pyr_idx][{1, {}, pyr_y, pyr_x}])
      end
    end
  end
  --print('back end')
  return gradInput

  --local dimension = self:_getPositiveDimension(input)

  ---- clear out invalid gradInputs
  --for i=#input+1, #self.gradInput do
  --  self.gradInput[i] = nil
  --end

  --local offset = 1
  --for i=1,#input do
  --  local currentOutput = input[i]
  --  local currentGradInput = gradOutput:narrow(dimension, offset,
  --  currentOutput:size(dimension))
  --  self.gradInput[i]:copy(currentGradInput)
  --  offset = offset + currentOutput:size(dimension)
  --end
end


function PyramidMultiplexer:type(type, tensorCache)
  self.gradInput = {}
  return parent.type(self, type, tensorCache)
end

--function PyramidMultiplexer:__init(dimension, nInputDims)
--  parent.__init(self)
--  self.size = torch.LongStorage()
--  self.dimension = dimension
--  self.gradInput = {}
--  self.nInputDims = nInputDims
--end
--function PyramidMultiplexer:_getPositiveDimension(input)
--  local dimension = self.dimension
--  if dimension < 0 then
--    dimension = input:dim() + dimension + 1
--  elseif self.nInputDims and input[1]:dim()==(self.nInputDims+1) then
--    dimension = dimension + 1
--  end
--  return dimension
--end
