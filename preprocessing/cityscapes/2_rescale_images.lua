require 'image'
local gm = require 'graphicsmagick'
local json = require 'json'

--local cx_start = 1
--local cx_end = 2048
--local cy_start = 26
----local cy_end = 826
----img_width = 1632
----img_height = 640
--local cy_end = 950
--local crop_width = cx_end - cx_start + 1
--local crop_height = cy_end - cy_start + 1
--local img_width = 1632
--local img_height = 736

--flags.DEFINE_integer('cx_start', 0, '')
--flags.DEFINE_integer('cx_end', 2048, '')
--flags.DEFINE_integer('cy_start', 30, '')
--flags.DEFINE_integer('cy_end', 894, '')
--local cx_start = 0
--local cx_end = 2047
--local cy_start = 30
--local cy_end = 893
--local img_width = 1024
--local img_height = 432

local cx_start = 96
local cx_end = 2047
local cy_start = 30
local cy_end = 893
local img_width = 1952
local img_height = 864
--local img_width = 1024
--local img_height = 448
--local img_width = 1504
--local img_height = 672
local rescale = false
local crop_width = cx_end - cx_start + 1
local crop_height = cy_end - cy_start + 1
print('Crop ratio = ', crop_width / crop_height)
--local img_width = crop_width
--local img_height = crop_height
print('Resize ratio = ', img_width / img_height)

--local root_dir = '/mnt/ikreso/datasets/Cityscapes/'
local root_dir = '/home/kivan/datasets/Cityscapes/'
img_dir = root_dir .. "/2048x1024/rgb/"
gt_dir = root_dir .. "/2048x1024/gt_rgb/"
depth_dir = root_dir .. '/2048x1024/depth/'
out_dir =  root_dir .. '/torch/' .. img_width .. 'x' .. img_height .. '/img/'
os.execute("mkdir -p " .. out_dir)

local function RescaleDataset(name)
  local img_subdir = img_dir .. name
  local gt_subdir = gt_dir .. name
  local rgb_out_subdir = out_dir .. '/data/' .. name
  local gt_out_subdir = out_dir .. '/labels/' .. name
  local img_list = {}
  os.execute('mkdir -p ' .. rgb_out_subdir)
  os.execute('mkdir -p ' .. gt_out_subdir)
  for dir in paths.iterdirs(img_subdir) do
    local seq_dir = img_subdir..'/'..dir..'/'
    local seq_gt_dir = gt_subdir..'/'..dir..'/'
    local seq_depth_dir = depth_dir .. name .. '/' .. dir .. '/'
    local seq_rgb_out_dir = rgb_out_subdir..'/'..dir..'/'
    local seq_gt_out_dir = gt_out_subdir..'/'..dir..'/'
    local seq_depth_out_dir = out_dir .. '/depth/' .. name .. '/' .. dir .. '/'
    os.execute('mkdir -p ' .. seq_rgb_out_dir)
    os.execute('mkdir -p ' .. seq_gt_out_dir)
    os.execute('mkdir -p ' .. seq_depth_out_dir)
    for rgb_img_name in paths.iterfiles(seq_dir) do
      print(rgb_img_name)
      local cam_path = root_dir .. 'camera_trainvaltest/' .. name .. '/' .. dir .. '/' ..
                       rgb_img_name:sub(1,-5) .. '_camera.json'
                       --rgb_img_name:sub(1,-17) .. '_camera.json'
      print(cam_path)
      local cam_params = json.load(cam_path)
      local name_prefix = rgb_img_name:sub(1,-5)
      table.insert(img_list, {name_prefix..'.ppm', dir .. '/' .. name_prefix..'.ppm', cam_params})
      local gt_img_name = name_prefix .. '.ppm'
      local gt_img = image.load(seq_gt_dir .. gt_img_name, 3, 'byte')
      gt_img = image.crop(gt_img, cx_start, cy_start, cx_end+1, cy_end+1)
      local rgb_img = gm.Image(seq_dir .. rgb_img_name)
      local depth_img = gm.Image(seq_depth_dir .. name_prefix .. '_leftImg8bit.png')
      depth_img:crop(crop_width, crop_height, cx_start, cy_start)
      rgb_img:crop(crop_width, crop_height, cx_start, cy_start)
      print(rgb_img:size())
      if rescale then
        gt_img = image.scale(gt_img, img_width, img_height, "simple")
        rgb_img:size(img_width, img_height)
        depth_img:size(img_width, img_height)
      end
      rgb_img:save(seq_rgb_out_dir .. name_prefix .. '.png')
      image.save(seq_gt_out_dir .. name_prefix .. '.png', gt_img)
      depth_img:save(seq_depth_out_dir .. name_prefix .. '.png')
    end
  end
  torch.save(root_dir .. name .. '_img_list.t7', img_list)
end

RescaleDataset('train')
RescaleDataset('val')
