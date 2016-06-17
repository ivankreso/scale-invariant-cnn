if isdefined_data_container_multifile then
  return
end
isdefined_data_container_multifile = true

require 'torch'
require 'dok'
require 'cudnn'
--require '../params'
paths.dofile('data_container_helper.lua')

math.randomseed(os.time())

local DataContainerMultiFile = torch.class('DataContainerMultiFile')

function DataContainerMultiFile:__init(...)
  local args = dok.unpack(
  {...},
  'InitializeData',
  'Initializes a DataContainer',
  {arg='data_dir', type='string', help='Path to data', req = true},
  --{arg='batch_size', type='number', help='Number of Elements in each Batch',req = true},
  {arg='prefix', type='string', help='Path to image list txt', req = true},
  {arg='tensor_type', type='string', help='Type of Tensor', default = 'torch.CudaTensor'},
  {arg='num_files', type='number', help='Size of epoch', default = 1}
  )

  self.num_files = args.num_files
  self.tensor_type = args.tensor_type
  self.prefix = args.prefix
  self.data_dir = args.data_dir .. '/' .. self.prefix .. '/'
  --self.epoch_size = math.floor(self.epoch_size * self.num_imgs / self.batch_size)

  self.batch_data = torch.Tensor():type(self.tensor_type)
  self.batch_labels = torch.Tensor():type(self.tensor_type)
  --local batch_data_sz = torch.LongStorage({self.batch_size, self.channels, img_height, img_width})
  --local batch_labels_sz = torch.LongStorage({self.batch_size, img_height, img_width})
  --self.batch_data = torch.Tensor(batch_data_sz):type(self.tensor_type)
  --self.batch_labels = torch.Tensor(batch_labels_sz):type(self.tensor_type)

  -- shuffle file order
  self.file_shuffle = torch.randperm(self.num_files)
  self.file_cnt = 0
  self.epoch_iter = 0
end

function DataContainerMultiFile:ReadNextFile()
  self:FreeMemory()
  self.file_cnt = self.file_cnt + 1
  if self.file_cnt > self.num_files then
    self.file_cnt = 0
    if self.prefix == 'train' then
      self.file_shuffle = torch.randperm(self.num_files)
      self:ReadNextFile()
    end
    return false
  end

  local prefix = self.data_dir .. self.file_shuffle[self.file_cnt] .. '_' .. self.prefix
  --print('Reading ... ' .. prefix)
  self.data = torch.load(prefix .. "_data.t7")
  self.labels = torch.load(prefix .. "_labels.t7")
  self.filenames = torch.load(prefix .. "_filenames.t7")
  --self.depth_data = torch.load(self.data_dir .. '/' .. self.prefix .. "_depth.t7")
  self.num_imgs = #self.data
  self.img_shuffle = torch.randperm(self.num_imgs)
  self.img_cnt = 0

  return true
end

function DataContainerMultiFile:GetNextBatch()
  if self.file_cnt == 0 or self.img_cnt >= self.num_imgs then
  --if self.file_cnt == 0 or self.img_cnt > 5 then
    if not self:ReadNextFile() then
      self.epoch_iter = 0
      return false
    end
    --return false
  end
  self.epoch_iter = self.epoch_iter + 1
  self.img_cnt = self.img_cnt + 1
  local img_num = self.img_shuffle[self.img_cnt]
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
  --self.batch_data = self.data[img_num]:view(1, self.channels, self.img_height, self.img_width):cuda()
  --self.batch_labels = self.labels[img_num][1]:view(1, self.img_height, self.img_width):cuda()
  --self.weights = self.labels[img_num][2]
  self.batch_data = SendPyramidToGPU(self.data[img_num])
  self.batch_labels = self.labels[img_num][1]:cuda()
  self.weights = self.labels[img_num][2]:cuda()

  return self.batch_data, self.batch_labels, self.weights, filename
end

function DataContainerMultiFile:size()
  if self.prefix == 'train' then
    return 2975
  else
    return 500
  end
end

function DataContainerMultiFile:PrintProgress()
  xlua.progress(self.epoch_iter, self:size())
end

function DataContainerMultiFile:FreeMemory()
  self.data = nil
  self.labels = nil
  self.filenames = nil
  collectgarbage()
end

