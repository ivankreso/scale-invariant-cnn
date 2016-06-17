paths.dofile('../../dataset_helper.lua')
paths.dofile('../../eval_helper.lua')
local gm = require 'graphicsmagick'
require 'image'

local class_info, class_color_map, num_classes = paths.dofile('class_colors.lua')

--local correct_width = 1632
--local correct_height = 736
--local correct_width = 1024
--local correct_height = 448
--local correct_width = 1504
--local correct_height = 672
local correct_width = 1024
local correct_height = 432
--local correct_width = 1952
--local correct_height = 864
local root_dir = '/home/kivan/datasets/Cityscapes/'
--local root_dir = '/mnt/ikreso/datasets/Cityscapes/'
local in_dir = root_dir .. correct_width .. 'x' .. correct_height .. '/'
local data_dir = in_dir .. '/img/'
local out_dir = in_dir .. '/torch/'

os.execute("mkdir -p " .. out_dir)

local function PrepareDataset(name, max_in_file)
  local rgb_subdir = data_dir .. '/data/' .. name
  local gt_subdir = data_dir .. '/labels/' .. name
  local save_dir = out_dir .. '/' .. name .. '/'
  os.execute('mkdir -p ' .. save_dir)
  local file_num = 1
  local data = {}
  local labels = {}
  local filenames = {}
  local rgb_files = {}
  local gt_files = {}
  local img_names = {}
  for dir in paths.iterdirs(rgb_subdir) do
    local seq_rgb_dir = rgb_subdir..'/'..dir..'/'
    local seq_gt_dir = gt_subdir..'/'..dir..'/'
    for img_name in paths.iterfiles(seq_rgb_dir) do
      table.insert(rgb_files, seq_rgb_dir .. img_name)
      table.insert(gt_files, seq_gt_dir .. img_name)
      table.insert(img_names, img_name)
    end
  end
  local idx_shuffle = torch.randperm(#img_names)

  --local class_hist = torch.IntTensor(num_classes):fill(0)
  local class_hist = torch.DoubleTensor(num_classes):fill(0)
  for i = 1, idx_shuffle:size(1) do
    local idx = idx_shuffle[i]
    local gt_img = image.load(gt_files[idx], 3, 'byte'):int()
    local rgb_img = image.load(rgb_files[idx])
    CountClassHistogram(gt_img, class_color_map, class_hist)
    xlua.progress(i, idx_shuffle:size(1))

    local num_channels = rgb_img:size(1)
    local img_height = rgb_img:size(2)
    local img_width = rgb_img:size(3)
    assert(correct_width == img_width and correct_height == img_height)
    local label_img = torch.IntTensor(img_height, img_width)
    local class_weights = ConvertLabelFormat(gt_img, class_color_map, num_classes, label_img)

    local norm_img = rgb_img:float()
    for c = 1, num_channels do
      norm_img[c] = (norm_img[c] - norm_img[c]:mean()) / norm_img[c]:std()
    end

    table.insert(data, norm_img:view(1, num_channels, img_height, img_width))
    --table.insert(labels, label_img:view(img_height, img_width))
    table.insert(labels, {label_img:view(1, img_height, img_width), class_weights})
    table.insert(filenames, img_names[idx])

    if i % max_in_file == 0 or i == idx_shuffle:size(1) then
      torch.save(save_dir .. file_num .. '_' .. name .. "_data.t7", data)
      torch.save(save_dir .. file_num .. '_' .. name .. "_labels.t7", labels)
      torch.save(save_dir .. file_num .. '_' .. name .. "_filenames.t7", filenames)        
      file_num = file_num + 1
      data = {}
      labels = {}
      filenames = {}
      collectgarbage()
    end
  end
  --print(class_hist)
  class_hist = class_hist:float() /  class_hist:sum() * 100
  for i = 1, class_hist:size(1) do
    io.write(string.format('%0.2f', Round(class_hist[i], 2)) .. ' & ')
  end
  print('')
end

print(class_info)
--torch.save(out_dir .. "/class_colors.t7", {class_color_map, class_info})

local max_in_file = 50
--local max_in_file = 300
PrepareDataset("train", max_in_file)
PrepareDataset("val", max_in_file)

-- bad
--local function PrepareDataset(name, max_in_file)
--  local rgb_subdir = data_dir .. '/data/' .. name
--  local gt_subdir = data_dir .. '/labels/' .. name
--  local num_imgs = 0
--  local file_num = 1
--  local data = {}
--  local labels = {}
--  local filenames = {}
--  print(rgb_subdir)
--  for dir in paths.iterdirs(rgb_subdir) do
--    local seq_rgb_dir = rgb_subdir..'/'..dir..'/'
--    local seq_gt_dir = gt_subdir..'/'..dir..'/'
--    for img_name in paths.iterfiles(seq_rgb_dir) do
--      --print(img_name)
--      local gt_img = image.load(seq_gt_dir .. img_name, 3, 'byte'):int()
--      local rgb_img = image.load(seq_rgb_dir .. img_name)
--      num_imgs = num_imgs + 1
--      xlua.progress(num_imgs, max_in_file)
--
--      local num_channels = rgb_img:size(1)
--      local img_height = rgb_img:size(2)
--      local img_width = rgb_img:size(3)
--      assert(correct_width == img_width and correct_height == img_height)
--      local label_img = torch.IntTensor(img_height, img_width)
--      ConvertLabelFormat(gt_img, class_color_map, label_img)
--
--      local norm_img = rgb_img:float()
--      for c = 1, num_channels do
--        norm_img[c] = (norm_img[c] - norm_img[c]:mean()) / norm_img[c]:std()
--      end
--
--      table.insert(data, norm_img:view(num_channels, img_height, img_width))
--      table.insert(labels, label_img:view(img_height, img_width))
--      table.insert(filenames, img_name)
--
--      if num_imgs == max_in_file then
--        torch.save(out_dir .. "/" .. file_num .. '_' .. name .. "_data.t7", data)
--        torch.save(out_dir .. "/" .. file_num .. '_' .. name .. "_labels.t7", labels)
--        torch.save(out_dir .. "/" .. file_num .. '_' .. name .. "_filenames.t7", filenames)        
--        num_imgs = 0
--        file_num = file_num + 1
--        data = {}
--        labels = {}
--        filenames = {}
--        collectgarbage()
--      end
--    end
--  end
--  if num_imgs > 0 and num_imgs ~= max_in_file then
--    torch.save(out_dir .. "/" .. file_num .. '_' .. name .. "_data.t7", data)
--    torch.save(out_dir .. "/" .. file_num .. '_' .. name .. "_labels.t7", labels)
--    torch.save(out_dir .. "/" .. file_num .. '_' .. name .. "_filenames.t7", filenames)        
--    num_imgs = 0
--    file_num = file_num + 1
--    data = {}
--    labels = {}
--    filenames = {}
--    collectgarbage()
--  end
--end


