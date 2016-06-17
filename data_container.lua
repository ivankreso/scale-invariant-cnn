if isdefined_data_container then
  return
end
isdefined_data_container = true

require 'torch'
require 'dok'
require 'cudnn'
--require '../params'


math.randomseed(os.time())

local DataContainer = torch.class('DataContainer')

function DataContainer:__init(...)
  local args = dok.unpack(
  {...},
  'InitializeData',
  'Initializes a DataContainer',
  {arg='data_dir', type='string', help='Path to data', req = true},
  {arg='batch_size', type='number', help='Number of Elements in each Batch',req = true},
  {arg='prefix', type='string', help='Path to image list txt', req = true},
  {arg='tensor_type', type='string', help='Type of Tensor', default = 'torch.CudaTensor'},
  {arg='epoch_size', type='number', help='Size of epoch', default = 1}
  )

  self.data_dir = args.data_dir
  self.batch_size = args.batch_size
  self.epoch_size = args.epoch_size
  self.tensor_type = args.tensor_type
  self.prefix = args.prefix
  self.batch_labels = torch.Tensor():type(self.tensor_type)
  self.current_img = 1
  self.data = torch.load(self.data_dir .. '/' .. self.prefix .. "_data.t7")
  self.depth_data = torch.load(self.data_dir .. '/' .. self.prefix .. "_depth.t7")
  self.filenames = torch.load(self.data_dir .. '/' .. self.prefix .. "_filenames.t7")
  self.labels = torch.load(self.data_dir .. '/' .. self.prefix .. "_labels.t7")
  self.num_imgs = #self.data
  --self.epoch_size = math.floor(self.epoch_size * self.num_imgs / self.batch_size)
  self.epoch_size = self.num_imgs
  self.batch_cnt = 0
  self.channels = self.data[1]:size(1)
  --self.img_height = self.data[1]:size(2)
  --self.img_width = self.data[1]:size(3)

  --local batch_data_sz = torch.LongStorage({self.batch_size, self.channels, img_height, img_width})
  --local batch_labels_sz = torch.LongStorage({self.batch_size, img_height, img_width})
  --self.batch_data = torch.Tensor(batch_data_sz):type(self.tensor_type)
  --self.batch_labels = torch.Tensor(batch_labels_sz):type(self.tensor_type)

  -- shuffle data
  self.shuffle = torch.randperm(self.num_imgs)
end

function DataContainer:GetNextBatch()
  if self.current_img > self.num_imgs then
    self.current_img = 1
    -- shuffle data again if training
    if self.prefix == 'train' then
      self.shuffle = torch.randperm(self.num_imgs)
    end
    return nil
  end
  local img_num = self.shuffle[self.current_img]
  --local img_num = self.current_img
  local filename = self.filenames[img_num]
  --print(self.batch_data:lt(0.1):sum())
  --require 'gnuplot'
  --gnuplot.figure()
  --gnuplot.hist(self.batch_data:view(-1), 100)
  --gnuplot.figure()
  --gnuplot.title('disparity')
  --gnuplot.hist(self.depth_data[img_num][1]:view(-1), 100)
  ----gnuplot.hist(self.data[img_num][3]:view(-1), 100)
  ----gnuplot.figure()
  ----gnuplot.hist(self.depth_data[img_num][2]:view(-1), 100)
  --io.read()

  --print(self.data[img_num][3]:view(-1))
  --self.batch_data = self.data[img_num]:view(1, self.channels, img_height, img_width):cuda()
  self.batch_data = self.data[img_num]:cuda()
  local height = self.batch_data:size(2)
  local width = self.batch_data:size(3)
  self.batch_data = self.batch_data:view(1, self.channels, height, width)
  self.batch_labels = self.labels[img_num][1]:view(1, height, width):cuda()
  self.weights = self.labels[img_num][2]
  local depth_img = self.depth_data[img_num][2]

  self.current_img = self.current_img + 1
  --self.current_img = self.current_img + 9

  return self.batch_data, self.batch_labels, self.weights, filename, depth_img
  --return self.batch_data, self.batch_labels, filename, depth_img
end

function DataContainer:PrintProgress()
  xlua.progress(self.current_img, self.num_imgs)
end

function DataContainer:size()
  return self.epoch_size * self.batch_size
end
