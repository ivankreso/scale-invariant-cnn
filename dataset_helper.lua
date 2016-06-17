function ConvertLabelFormat(gt_img, color_map, num_classes, label_map)
  gt_img = gt_img:transpose(1,3):transpose(1,2):contiguous()
  local rgb_ptr = torch.data(gt_img:contiguous())
  local dst_ptr = torch.data(label_map:contiguous())
  local class_hist = torch.IntTensor(num_classes):fill(0)
  local class_hist_ptr = torch.data(class_hist:contiguous())
  local height = label_map:size(1)
  local width = label_map:size(2)
  for i = 0, height-1 do
    local dst_stride = i * width
    local rgb_stride = dst_stride * 3
    for j = 0, width-1 do
      local stride = rgb_stride + j*3
      local r = rgb_ptr[stride]
      local g = rgb_ptr[stride + 1]
      local b = rgb_ptr[stride + 2]
      local key = string.format("%03d", r) .. string.format("%03d", g) .. string.format("%03d", b)
      --print(gt_img[{i+1,j+1,{}}])
      local id = color_map[key][1]
      assert(id >= 0 and id <= num_classes)
      dst_ptr[dst_stride + j] = id
      if id > 0 then
        class_hist_ptr[id-1] = class_hist_ptr[id-1] + 1
      end
    end
  end
  class_hist = class_hist:float()
  local num_labels = class_hist:sum()
  -- the best 1e-2 - 1e-3
  --local clip_p = 1e-2
  local clip_p = 1e-3
  --local clip_p = 1e-1
  for i = 1, num_classes do
    local p = class_hist[i] / num_labels
    if p > 1e-7 then
      if p < clip_p then
        p = clip_p
      end
      class_hist[i] = 1.0 / p
    else
      class_hist[i] = 0
    end
  end
  --print(class_hist)
  return class_hist
end


function CountClassHistogram(rgb_img, color_map, class_hist)
  local height = rgb_img:size(2)
  local width = rgb_img:size(3)
  local num_classes = class_hist:size(1)
  local gt_img = rgb_img:transpose(1,3):transpose(1,2):contiguous()
  local rgb_ptr = torch.data(gt_img:contiguous())
  local class_hist_ptr = torch.data(class_hist:contiguous())
  for i = 0, height-1 do
    local dst_stride = i * width
    local rgb_stride = dst_stride * 3
    for j = 0, width-1 do
      local stride = rgb_stride + j*3
      local r = rgb_ptr[stride]
      local g = rgb_ptr[stride + 1]
      local b = rgb_ptr[stride + 2]
      local key = string.format("%03d", r) .. string.format("%03d", g) .. string.format("%03d", b)
      --print(gt_img[{i+1,j+1,{}}])
      local id = color_map[key][1]
      assert(id >= 0 and id <= num_classes)
      if id > 0 then
        class_hist_ptr[id-1] = class_hist_ptr[id-1] + 1
      end
    end
  end
end

--function ConvertLabelFormat(gt_img, color_map, num_classes, label_map)
--  gt_img = gt_img:transpose(1,3):transpose(1,2):contiguous()
--  local rgb_ptr = torch.data(gt_img:contiguous())
--  local dst_ptr = torch.data(label_map:contiguous())
--  local class_hist = torch.IntTensor(num_classes):fill(0)
--  local class_hist_ptr = torch.data(class_hist:contiguous())
--  local height = label_map:size(1)
--  local width = label_map:size(2)
--  for i = 0, height-1 do
--    local dst_stride = i * width
--    local rgb_stride = dst_stride * 3
--    for j = 0, width-1 do
--      local stride = rgb_stride + j*3
--      local r = rgb_ptr[stride]
--      local g = rgb_ptr[stride + 1]
--      local b = rgb_ptr[stride + 2]
--      local key = string.format("%03d", r) .. string.format("%03d", g) .. string.format("%03d", b)
--      --print(gt_img[{i+1,j+1,{}}])
--      local id = color_map[key][1]
--      assert(id >= 0 and id <= num_classes)
--      dst_ptr[dst_stride + j] = id
--      if id > 0 then
--        class_hist_ptr[id-1] = class_hist_ptr[id-1] + 1
--      end
--    end
--  end
--end


function GetSubmissionFormat(color_img, color_map, num_classes)
  local rgb_img = color_img:transpose(1,3):transpose(1,2):contiguous()
  local height = rgb_img:size(1)
  local width = rgb_img:size(2)
  local gray_img = torch.ByteTensor(1, height, width):fill(0)

  local rgb_ptr = torch.data(rgb_img:contiguous())
  local gray_ptr = torch.data(gray_img:contiguous())
  for i = 0, height-1 do
    local gray_stride = i * width
    local rgb_stride = gray_stride * 3
    for j = 0, width-1 do
      local stride = rgb_stride + j*3
      local r = rgb_ptr[stride]
      local g = rgb_ptr[stride + 1]
      local b = rgb_ptr[stride + 2]
      local key = string.format("%03d", r) .. string.format("%03d", g) .. string.format("%03d", b)
      --print(gt_img[{i+1,j+1,{}}])
      local id = color_map[key][3]
      --print(key, id)
      assert(id >= 0 and id <= 33)
      gray_ptr[gray_stride + j] = id
    end
  end
  return gray_img
end


local function ConvertLabelFormatSlow(gt_img, color_map, label_img)
  for i = 1, label_img:size(1) do
    for j = 1, label_img:size(2) do
      --print(gt_img[{{},i,j}])
      local r = gt_img[{1,i,j}]
      local g = gt_img[{2,i,j}]
      local b = gt_img[{3,i,j}]
      local key = string.format("%03d", r) .. string.format("%03d", g) .. string.format("%03d", b)
      local id = color_map[key][1]
      assert(id ~= nil)
      label_img[{i,j}] = id
    end
  end
end

function PrecomputeScaleRouting(depth_img, baseline, filename, debug_save_dir)
  local function GetPyramidPosition(sf)
    local pyr_pos = #downsample_factors
    for k = 1, #downsample_factors do
      local df = downsample_factors[k]
      if sf <= df then
        if k == 1 then
          pyr_pos = k
          break
        else
          local prev_df = downsample_factors[k-1]
          local dist1 = math.abs(sf-prev_df)
          local dist2 = math.abs(sf-df)
          if dist1 < dist2 then
            pyr_pos = k-1
          else
            pyr_pos = k
          end
          break
        end
      end
    end
    return pyr_pos
  end
  local function GetLocationInPyramid(pyr_pos, x, y)
    --local px = torch.round((x / first_width) * img_sizes[pyr_pos][1])
    --local py = torch.round((y / first_height) * img_sizes[pyr_pos][2])
    --local px = torch.round((x / first_width) * net_out_sizes[pyr_pos][1])
    --local py = torch.round((y / first_height) * net_out_sizes[pyr_pos][2])
    local px = torch.round((x / target_width) * net_out_sizes[pyr_pos][1])
    local py = torch.round((y / target_height) * net_out_sizes[pyr_pos][2])
    --print((y / first_height) * img_sizes[pyr_pos][2])
    if px < 1 then
      px = 1
    end
    if py < 1 then
      py = 1
    end
    return px, py
  end

  local height = depth_img:size(1)
  local width = depth_img:size(2)
  local debug_img = {}
  if debug_save_dir ~= nil then
    for i = 1, #scales do
      table.insert(debug_img, torch.ByteTensor(3, height, width):fill(0))
    end
  end

  local scale_routing = {}
  for i = 1, #scales do
    table.insert(scale_routing, torch.IntTensor(height, width, 3):fill(0))
  end
  for i = 1, height do
    for j = 1, width do
      local d = depth_img[{i,j}]
      --print(d)
      for k = 1, #scales do
        local sf = (d * scales[k] / baseline) / rf_size
        --print(sf)
        local pyr_pos = GetPyramidPosition(sf)
        --print('pos = ', pyr_pos)
        --print(j,i)
        local x, y = GetLocationInPyramid(pyr_pos, j, i)
        --label_pyr[pyr_pos][{y,x}] = label_img[{i,j}]
        --print(pyr_pos, x,y)
        scale_routing[k][{i,j,1}] = pyr_pos
        scale_routing[k][{i,j,2}] = y
        scale_routing[k][{i,j,3}] = x
        if debug_save_dir ~= nil then
          for c = 1, 3 do
            debug_img[k][{c,i,j}] = color_coding[pyr_pos][c]
          end
        end
      end
    end
  end
  if debug_save_dir ~= nil then
    for i = 1, #scales do
      image.save(debug_save_dir .. filename:sub(1,-5) .. '_' .. i .. '.png', debug_img[i])
    end
  end
  return scale_routing
end


function ComputePyramidResolutions(width, height, sfactor, num_of_scales)
  local function AlignForPooling(num_pixels)
    local sub_factor = 16
    local res = num_pixels % sub_factor
    if res >= (sub_factor/2) then
      res = -(sub_factor - res)
    end
    num_pixels = num_pixels - res
    return num_pixels
  end
  local sizes = {{width, height}}
  local aspect_ratio = width / height
  prev_w = width
  prev_h = height
  print(width .. 'x' .. height)
  for i = 1, num_of_scales-1 do
    local new_w = torch.round(prev_w / sfactor)
    new_w = AlignForPooling(new_w)
    --local new_h = torch.round(prev_h / sfactor)
    local new_h = torch.round(new_w / aspect_ratio)
    new_h = AlignForPooling(new_h)
    table.insert(sizes, {new_w, new_h})
    prev_w = new_w
    prev_h = new_h
    print(new_w .. 'x' .. new_h)
  end
  return sizes
end


function PrecomputeScaleRoutingBaseline(depth_img, baseline, filename, debug_save_dir)
  local function GetLocationInPyramid(pyr_pos, x, y)
    --local px = torch.round((x / first_width) * img_sizes[pyr_pos][1])
    --local py = torch.round((y / first_height) * img_sizes[pyr_pos][2])
    --local px = torch.round((x / first_width) * net_out_sizes[pyr_pos][1])
    --local py = torch.round((y / first_height) * net_out_sizes[pyr_pos][2])
    local px = torch.round((x / target_width) * net_out_sizes[pyr_pos][1])
    local py = torch.round((y / target_height) * net_out_sizes[pyr_pos][2])
    --print((y / first_height) * img_sizes[pyr_pos][2])
    if px < 1 then
      px = 1
    end
    if py < 1 then
      py = 1
    end
    return px, py
  end

  local height = depth_img:size(1)
  local width = depth_img:size(2)
  local debug_img = {}
  if debug_save_dir ~= nil then
    for i = 1, #scales do
      table.insert(debug_img, torch.ByteTensor(3, height, width):fill(0))
    end
  end

  local scale_routing = {}
  for i = 1, #scales do
    table.insert(scale_routing, torch.IntTensor(height, width, 3):fill(0))
  end
  for i = 1, height do
    for j = 1, width do
      local d = depth_img[{i,j}]
      --print(d)
      for k = 1, #scales do
        local sf = (d * scales[k] / baseline) / rf_size
        --print(sf)
        local pyr_pos = k
        --print('pos = ', pyr_pos)
        --print(j,i)
        local x, y = GetLocationInPyramid(pyr_pos, j, i)
        --label_pyr[pyr_pos][{y,x}] = label_img[{i,j}]
        --print(pyr_pos, x,y)
        scale_routing[k][{i,j,1}] = pyr_pos
        scale_routing[k][{i,j,2}] = y
        scale_routing[k][{i,j,3}] = x
        if debug_save_dir ~= nil then
          for c = 1, 3 do
            debug_img[k][{c,i,j}] = color_coding[pyr_pos][c]
          end
        end
      end
    end
  end
  if debug_save_dir ~= nil then
    for i = 1, #scales do
      image.save(debug_save_dir .. filename:sub(1,-5) .. '_' .. i .. '.png', debug_img[i])
    end
  end
  return scale_routing
end
