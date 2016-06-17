require 'image'

class_colors, class_color_map = paths.dofile('class_colors.lua')

gt_dir = '/home/kivan/datasets/Cityscapes/gtFine_trainvaltest/gtFine/'
img_dir = '/home/kivan/datasets/Cityscapes/leftImg8bit_trainvaltest/leftImg8bit/'
out_dir = '/home/kivan/datasets/Cityscapes/gtFine_trainvaltest/gtFine_19/'

local function ClearColors(img_path)
  local img = image.load(img_path, 3, 'byte')
  local h = img:size(2)
  local w = img:size(3)
  for y = 1, h do
    for x = 1, w do
      local r = img[{1,y,x}]
      local g = img[{2,y,x}]
      local b = img[{3,y,x}]
      local key = string.format("%03d", r) .. string.format("%03d", g) .. string.format("%03d", b)
      if class_color_map[key] == nil then
        img[{{},y,x}]:fill(0)
      end
    end
  end
  return img
end

local function CreateGT(name)
  local img_subdir = img_dir..name
  local gt_subdir = gt_dir..name
  local out_subdir = out_dir..name
  os.execute('mkdir -p '..out_subdir)
  for dir in paths.iterdirs(img_subdir) do
    local seq_dir = img_subdir..'/'..dir..'/'
    local seq_gt_dir = gt_subdir..'/'..dir..'/'
    local seq_out_dir = out_subdir..'/'..dir..'/'
    os.execute('mkdir -p '..seq_out_dir)
    for img_name in paths.iterfiles(seq_dir) do
      local gt_img_name = img_name:sub(1,-17)..'_gtFine_color.png'
      print(gt_img_name)
      local gt_img = ClearColors(seq_gt_dir .. gt_img_name)
      image.save(seq_out_dir .. gt_img_name, gt_img)
    end
  end
end

CreateGT('train')
CreateGT('val')
