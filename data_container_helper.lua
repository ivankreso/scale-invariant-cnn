

function SendPyramidToGPU(data)
  local x = {}
  local size = #data
  for i = 1, size-1 do
    table.insert(x, data[i]:cuda())
  end
  table.insert(x, data[size])
  --local routing = {}
  --for i = 1, #data[size] do
  --  table.insert(routing, data[size][i]:cuda())
  --end
  --table.insert(x, routing)
  return x
end

function SendPyramidToGPUBaseline(data)
  local x = {}
  local size = #data
  --print(size)
  table.insert(x, data[1]:cuda())
  table.insert(x, data[4]:cuda())
  table.insert(x, data[8]:cuda())
  --table.insert(x, data[size])
  local routing = {}
  for i = 1, #data[size] do
    table.insert(routing, data[size][i]:cuda())
  end
  table.insert(x, routing)
  return x
end
