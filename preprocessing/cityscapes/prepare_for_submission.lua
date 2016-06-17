require 'image'
require '../../dataset_helper.lua'

local function ConvertColorToGray(rgb_dir, out_dir, color_map, num_classes)
  for img_name in paths.iterfiles(rgb_dir) do
    print(img_name)
    local rgb_img = image.load(rgb_dir .. img_name, 3, 'byte')
    local gray_img = GetSubmissionFormat(rgb_img, color_map, num_classes)
    image.save(out_dir .. img_name, gray_img)
  end
end

local class_info, color_map, num_classes = paths.dofile('class_colors.lua')

local in_dir = '/mnt/ikreso/datasets/Cityscapes/pyramid/2048x1024_8/results/test/'
local out_dir = '/mnt/ikreso/datasets/Cityscapes/pyramid/2048x1024_8/results/test_submit/'
os.execute('mkdir -p ' .. out_dir)
ConvertColorToGray(in_dir, out_dir, color_map, num_classes)
