require 'image'
paths.dofile('../../dataset_helper.lua')
local gm = require 'graphicsmagick'

local num_classes = 11
--local img_width = 1216
--local img_height = 352
local img_width = 1248
local img_height = 384

--local img_width = 1856
--local img_height = 544

--local img_width = 768
--local img_height = 224

--local in_dir = "/home/kivan/datasets/KITTI/segmentation/semantic_segmentation/1_german_ros/"
local in_dir = "/home/kivan/datasets/KITTI/segmentation/semantic_segmentation/union/"
local out_dir = in_dir .. "/torch/" .. img_width .. 'x' .. img_height .. '/'
os.execute("mkdir -p " .. out_dir)

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
    xlua.progress(i, #imglist)
    --local img = image.load(img_dir .. "/data/rgb/" .. imglist[i], 3, 'byte')
    --local img = image.load(img_dir .. "/data/rgb/" .. imglist[i])
    local img = gm.Image(img_dir .. "/data/rgb/" .. imglist[i])
    --local scale_factor = img:size(1) / img_width
    local gt_img = image.load(img_dir .. "/labels/" .. imglist[i], 3, 'byte')
    local scaled_img = img:size(img_width, img_height)
    local scaled_gt_img = image.scale(gt_img, img_width, img_height, "simple")
    --image.save(rgb_dir .. imglist[i], scaled_img)
    scaled_img:save(rgb_dir .. imglist[i])
    image.save(gt_dir .. imglist[i], scaled_gt_img)

    local label_img = torch.IntTensor(img_height, img_width)
    local class_weights = ConvertLabelFormat(scaled_gt_img, label_colors, num_classes, label_img)

    local norm_img = scaled_img:toTensor('byte', 'RGB', 'DHW'):float()
    local num_channels = norm_img:size(1)
    for c = 1, num_channels do
      norm_img[c] = (norm_img[c] - norm_img[c]:mean()) / norm_img[c]:std()
    end
    --table.insert(data, norm_img:view(1, num_channels, img_height, img_width))
    --table.insert(disp_data, gm.load(disp_dir .. imglist[i], 'byte')[1])
    --print(imglist[i])

    table.insert(data, norm_img:view(num_channels, img_height, img_width))
    table.insert(labels, {label_img:view(img_height, img_width), class_weights})
    --table.insert(labels, label_img:view(img_height, img_width))

    local depth_img = gm.load(img_dir .. "/data/depth/" .. imglist[i], 'byte')[1]
    depth_img = image.scale(depth_img, img_width, img_height, "bilinear")
    depth_img = depth_img:view(img_height, img_width):float()
    --depth_img:mul(scale_factor)
    ----print(depth_img:size())
    ----print(depth_img)
    local norm_depth = (depth_img - depth_img:mean()) / depth_img:std()
    table.insert(depth, {norm_depth, depth_img})
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
