require 'image'
paths.dofile('../../dataset_helper.lua')
local gm = require 'graphicsmagick'

local num_classes = 11
local baseline = 0.5372
-- orig: 1241x376
local orig_width = 1241
local orig_height = 376

local img_sizes = {{1248, 384}, {1056, 320}, {880, 272}, {736, 224}, {624, 192}, {528, 160},
                   {448, 144}, {368, 112}, {320, 96}}
--local colors = {{255,0,0}, {}}
local color_coding = {{0,0,0}, {128,64,128}, {244,35,232}, {70,70,70}, {102,102,156}, {190,153,153},
{153,153,153}, {250,170,30}, {220,220,0}}

local downsample_factors = {}
for i = 1, #img_sizes do
  table.insert(downsample_factors, img_sizes[1][1] / img_sizes[i][1])
end
print(downsample_factors)

--local scale_size = 4
--local scale_size = 6
--local scale_size = 3
local scale_size = 2
local rf_size = 186

-- meters, rec field
--local scales = {{1, 100}, {2, 150}, {4, 200}}
local first_width = img_sizes[1][1]
local first_height = img_sizes[1][2]
local disp_scale = first_width / orig_width

local function SetPyramidLevels(pyr, depth)
  local disp_img = pyr[1]:clone()
end

--local img_width = 1856
--local img_height = 544

--local img_width = 768
--local img_height = 224

--local in_dir = "/home/kivan/datasets/KITTI/segmentation/semantic_segmentation/1_german_ros/"
local in_dir = "/home/kivan/datasets/KITTI/semantic_segmentation/"
local out_dir = in_dir .. "/torch/8_scales/"
local debug_save_dir = out_dir .. '/img/debug/'
os.execute("mkdir -p " .. debug_save_dir)
os.execute("mkdir -p " .. out_dir)
os.execute("mkdir -p " .. out_dir .. '/img/train/rgb')
os.execute("mkdir -p " .. out_dir .. '/img/valid/rgb')

local function CreateLabelPyramid(label_img, depth_img, filename)
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
    local px = torch.round((x / first_width) * img_sizes[pyr_pos][1])
    local py = torch.round((y / first_height) * img_sizes[pyr_pos][2])
    --print((y / first_height) * img_sizes[pyr_pos][2])
    if px < 1 then
      px = 1
    end
    if py < 1 then
      py = 1
    end
    return px, py
  end

  local height = label_img:size(1)
  local width = label_img:size(2)
  local label_pyr = {}
  for j = 1, #img_sizes do
    local w = img_sizes[j][1]
    local h = img_sizes[j][2]
    table.insert(label_pyr, torch.IntTensor(h, w):fill(0))
  end

  local debug_img = torch.ByteTensor(3, height, width):fill(0)
  for i = 1, height do
    for j = 1, width do
      local d = depth_img[{i,j}]
      --print(d)
      local sf = (d * scale_size / baseline) / rf_size
      --print(sf)
      local pyr_pos = GetPyramidPosition(sf)
      --print('pos = ', pyr_pos)
      --print(j,i)
      local x, y = GetLocationInPyramid(pyr_pos, j, i)
      --print(x,y)
      label_pyr[pyr_pos][{y,x}] = label_img[{i,j}]
      for k = 1, 3 do
        debug_img[{k,i,j}] = color_coding[pyr_pos][k]
      end
    end
  end
  image.save(debug_save_dir .. filename, debug_img)
  return label_pyr
end

local function PrepareDataset(img_dir, out_dir, name, label_colors)
  local rgb_dir = out_dir .. '/img/' .. name .. '/rgb/'
  local gt_dir = out_dir .. '/img/' .. name .. '/labels/'
  os.execute("mkdir -p " .. rgb_dir)
  os.execute("mkdir -p " .. gt_dir)

  local file = torch.DiskFile(img_dir .. "/img_list.txt", "r", true)
  print("Loading the images...")
  local imglist = {}
  while true do
    local filename = file:readString("*l")
    if file:hasError() then
      break
    else
      table.insert(imglist, filename)
    end
  end

  local data = {}
  local depth = {}
  local labels = {}
  for i = 1, #imglist do
    --print(imglist[i])
    local pyramid = {}
    xlua.progress(i, #imglist)
    --local img = image.load(img_dir .. "/data/rgb/" .. imglist[i], 3, 'byte')
    local img = gm.Image(img_dir .. "/data/rgb/" .. imglist[i])
    local gt_img = image.load(img_dir .. "/labels/" .. imglist[i], 3, 'byte')
    local scaled_gt_img = image.scale(gt_img, first_width, first_height, "simple")
    --local depth_img = gm.load(img_dir .. "/data/depth/" .. imglist[i], 'byte')[1]
    --local depth_img = gm.Image(img_dir .. "/data/depth/" .. imglist[i], 'byte')
    --local depth_img = gm.Image(img_dir .. "/data/depth/" .. imglist[i])
    local depth_img = gm.Image(img_dir .. "/data/depth/" .. imglist[i])
    depth_img:size(first_width, first_height)
    depth_img = depth_img:toTensor('byte','RGB','DHW'):float()[1]
    depth_img:mul(disp_scale)
    --print(depth_img[200])
    --depth_img = image.scale(depth_img, first_width, first_height, "bicubic")

    local norm_depth = (depth_img - depth_img:mean()) / depth_img:std()
    table.insert(depth, {norm_depth, depth_img})
    local label_img = torch.IntTensor(first_height, first_width)
    local class_weights = ConvertLabelFormat(scaled_gt_img, label_colors, num_classes, label_img)
    --table.insert(labels, label_img:view(first_height, first_width))
    local label_pyramid = CreateLabelPyramid(label_img, depth_img, imglist[i])

    for j = 1, #img_sizes do
      --local scaled_img = image.scale(img, img_width, img_height, "bicubic")
      local scaled_img = img:clone()
      local width = img_sizes[j][1]
      local height = img_sizes[j][2]
      scaled_img:size(width, height)
      --image.save(rgb_dir .. imglist[i], scaled_img)
      --scaled_img:save(rgb_dir .. imglist[i]:sub(1,-5) .. '_' .. j .. '.png')
      --image.save(gt_dir .. imglist[i], scaled_gt_img)
      local norm_img = scaled_img:toTensor('byte', 'RGB', 'DHW'):float()
      local num_channels = norm_img:size(1)
      for c = 1, num_channels do
        norm_img[c] = (norm_img[c] - norm_img[c]:mean()) / norm_img[c]:std()
      end
      --table.insert(data, norm_img:view(1, num_channels, img_height, img_width))
      --table.insert(disp_data, gm.load(disp_dir .. imglist[i], 'byte')[1])
      --print(imglist[i])
      table.insert(data, norm_img:view(num_channels, height, width))
      table.insert(labels, label_pyramid[j]:view(height, width))
    end

  end
  torch.save(out_dir .. "/" .. name .. "_data.t7", data)
  torch.save(out_dir .. "/" .. name .. "_depth.t7", depth)
  torch.save(out_dir .. "/" .. name .. "_labels.t7", labels)
  torch.save(out_dir .. "/" .. name .. "_filenames.t7", imglist)
end

local file = torch.DiskFile(in_dir .. "train/class_colors.txt", "r", true)
local label_colors = {}
local label_nums = {}
local cnt = 1
while true do
  local class_name = file:readString("*l")
  local r = file:readInt()
  local g = file:readInt()
  local b = file:readInt()
  if file:hasError() then
    break
  else
    --print(class_name .. r .. g .. b)
    local key = string.format("%03d", r) .. string.format("%03d", g) .. string.format("%03d", b)
    if class_name == "ignore" then
      label_colors[key] = {0, class_name}
      label_nums[0] = {r, g, b, class_name}
    else
      label_colors[key] = {cnt, class_name}
      label_nums[cnt] = {r, g, b, class_name}
      cnt = cnt + 1
    end
  end
end
print(label_nums)
torch.save(out_dir .. "/label_colors.t7", {label_colors, label_nums})

local data_dir = in_dir .. "/train/"
PrepareDataset(data_dir, out_dir, "train", label_colors)

data_dir = in_dir .. "/valid/"
PrepareDataset(data_dir, out_dir, "valid", label_colors)
