require 'image'
local gm = require 'graphicsmagick'
local json = require 'json'

local root_dir = '/home/kivan/datasets/Cityscapes/torch/1952x864/img/'
local img_dir = root_dir .. "/data/"
local gt_dir = root_dir .. "/gtFine_trainvaltest/gtFine_19/"
local depth_dir = root_dir .. "/depth/"
local out_dir =  root_dir .. '2048x1024/img/'
local camera_dir = '/home/kivan/datasets/Cityscapes/camera_trainvaltest/'
--os.execute("mkdir -p " .. out_dir)

local function RenameDataset(name)
  local img_subdir = img_dir .. name
  local gt_subdir = gt_dir .. name
  local rgb_out_subdir = out_dir .. '/data/' .. name
  local gt_out_subdir = out_dir .. '/labels/' .. name
  local img_list = {}
  --os.execute('mkdir -p ' .. rgb_out_subdir)
  --os.execute('mkdir -p ' .. gt_out_subdir)
  for dir in paths.iterdirs(img_subdir) do
    local seq_dir = img_subdir..'/'..dir..'/'
    local seq_gt_dir = gt_subdir..'/'..dir..'/'
    local seq_depth_dir = root_dir .. '/depth/' .. name .. '/' .. dir .. '/'
    local seq_rgb_out_dir = rgb_out_subdir..'/'..dir..'/'
    local seq_gt_out_dir = gt_out_subdir..'/'..dir..'/'
    local seq_depth_out_dir = out_dir .. '/depth/' .. name .. '/' .. dir .. '/'
    --os.execute('mkdir -p ' .. seq_rgb_out_dir)
    --os.execute('mkdir -p ' .. seq_gt_out_dir)
    --os.execute('mkdir -p ' .. seq_depth_out_dir)
    for rgb_img_name in paths.iterfiles(seq_dir) do
      print(rgb_img_name)
      local name_prefix = rgb_img_name:sub(1,-5)
      local cam_path = camera_dir .. name .. '/' .. dir .. '/' ..
                       name_prefix .. '_camera.json'
      local cam_params = json.load(cam_path)
      print(cam_params)
      table.insert(img_list, {name_prefix..'.png', dir .. '/' .. name_prefix..'.png', cam_params})
      --local gt_img_name = name_prefix .. '_gtFine_color.png'
      ----local gt_img = image.load(seq_gt_dir .. gt_img_name, 3, 'byte')
      ----gt_img = image.crop(gt_img, cx_start, cy_start, cx_end+1, cy_end+1)
      ----gt_img = image.scale(gt_img, img_width, img_height, "simple")
      --os.execute('cp ' .. seq_gt_dir .. gt_img_name .. ' '
      --           .. seq_gt_out_dir .. name_prefix .. '.png')
      --os.execute('cp ' .. seq_dir .. rgb_img_name .. ' ' .. seq_rgb_out_dir .. name_prefix .. '.png')
      --os.execute('cp ' .. seq_depth_dir .. rgb_img_name .. ' ' .. seq_depth_out_dir
      --           .. name_prefix .. '.png')
      --local rgb_img = gm.Image(seq_dir .. rgb_img_name)
      --local depth_img = gm.Image(seq_depth_dir .. rgb_img_name)
      --depth_img:crop(crop_width, crop_height, cx_start, cy_start)
      --depth_img:size(img_width, img_height)
      --rgb_img:crop(crop_width, crop_height, cx_start, cy_start)
      --rgb_img:size(img_width, img_height)
      --rgb_img:save(seq_rgb_out_dir .. name_prefix .. '.png')
      --image.save(seq_gt_out_dir .. name_prefix .. '.png', gt_img)
      --depth_img:save(seq_depth_out_dir .. name_prefix .. '.png')
    end
  end
  torch.save(root_dir .. name .. '_img_list.t7', img_list)
end

RenameDataset('train')
RenameDataset('val')
RenameDataset('test')
