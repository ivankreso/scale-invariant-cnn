if isdefined_data_container_multifile_threaded_test then
  return
end
isdefined_data_container_multifile_threaded_test = true

require 'torch'
require 'dok'
require 'cudnn'
local threads = require 'threads'
paths.dofile('data_container_helper.lua')
--require 'parallel'

--require '../params'
local data_dir
local prefix
local num_files
local file_cnt
local file_shuffle
local next_data
local next_filenames
local pool
local is_last_file = false

local function PrepareNextFileJob()
  local name_prefix = data_dir .. file_shuffle[file_cnt] .. '_' .. prefix
  --print('Reading ... ' .. name_prefix)
  local data = torch.load(name_prefix .. "_data.t7")
  local filenames = torch.load(name_prefix .. "_filenames.t7")
  return data, filenames
end
local function EndCallback(data, filenames)
  next_data = data
  next_filenames = filenames
  -- for debugging
  --print(file_cnt)
  --if file_cnt % 3 == 0 then
  --  is_last_file = true
  --end
  file_cnt = file_cnt + 1
  if file_cnt > num_files then
    file_shuffle = torch.randperm(num_files)
    file_cnt = 1
    is_last_file = true
  end
  --print('Done reading!')
end
local function PrepareNextFile()
  --self.fork = parallel.fork()
  --self.fork:exec(self.PrepareNextFileJob)
  --pool = threads.Threads(1,
  -- function(threadid)
  --    print('starting a new thread/state number ' .. threadid)
  --    --param_prefix = prefix -- get it the msg upvalue and store it in thread state
  -- end
  --)
  pool = threads.Threads(1)
  pool:addjob(PrepareNextFileJob, EndCallback)
end

math.randomseed(os.time())

local DataContainerMultiFileThreadedTest = torch.class('DataContainerMultiFileThreadedTest')

function DataContainerMultiFileThreadedTest:__init(...)
  local args = dok.unpack(
  {...},
  'InitializeData',
  'Initializes a DataContainerMultiFileThreadedTest',
  {arg='data_dir', type='string', help='Path to data', req = true},
  --{arg='batch_size', type='number', help='Number of Elements in each Batch',req = true},
  {arg='prefix', type='string', help='Path to image list txt', req = true},
  {arg='tensor_type', type='string', help='Type of Tensor', default = 'torch.CudaTensor'},
  {arg='num_files', type='number', help='Size of epoch', default = 1}
  )

  num_files = args.num_files
  self.tensor_type = args.tensor_type
  prefix = args.prefix
  data_dir = args.data_dir .. '/' .. prefix .. '/'
  --self.epoch_size = math.floor(self.epoch_size * self.num_imgs / self.batch_size)

  self.batch_data = torch.Tensor():type(self.tensor_type)
  --local batch_data_sz = torch.LongStorage({self.batch_size, self.channels, img_height, img_width})
  --self.batch_data = torch.Tensor(batch_data_sz):type(self.tensor_type)

  -- shuffle file order
  self.epoch_ended = false
  file_shuffle = torch.randperm(num_files)
  file_cnt = 1
  self.epoch_iter = 0
  self.num_imgs = -1
  self.img_cnt = -1
  PrepareNextFile()
  self:ReadNextFile()
  --pool:synchronize()
end

function DataContainerMultiFileThreadedTest:ReadNextFile()
  pool:terminate()

  self.data = next_data
  self.filenames = next_filenames
  collectgarbage()
  self.num_imgs = #self.data
  self.img_cnt = 0
  self.img_shuffle = torch.randperm(self.num_imgs)

  PrepareNextFile()
end

function DataContainerMultiFileThreadedTest:GetNextBatch()
  if self.img_cnt == self.num_imgs then
  --if self.img_cnt == 50 then
    self:ReadNextFile()
    if self.epoch_ended then
      self.epoch_ended = false
      self.epoch_iter = 0
      return false
    end
  end
  self.epoch_iter = self.epoch_iter + 1
  if self.resume_epoch == true then
    self.epoch_iter = self.epoch_iter - 1
    self.resume_epoch = false
  end

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
  if type(self.data[img_num]) == 'table' then
    self.batch_data = SendPyramidToGPU(self.data[img_num])
  else
    self.batch_data = self.data[img_num]:cuda()
  end

  -- if this is a final batch in epoch
  if self.img_cnt == self.num_imgs and is_last_file then
  --if self.img_cnt == 50 and is_last_file then
    self.epoch_ended = true
    is_last_file = false
    self:FreeMemory()
    --self.img_cnt = -1
  end

  return self.batch_data, nil, nil, filename
end


function DataContainerMultiFileThreadedTest:size()
  return 1525
end

function DataContainerMultiFileThreadedTest:PrintProgress()
  xlua.progress(self.epoch_iter, self:size())
end

function DataContainerMultiFileThreadedTest:FreeMemory()
  self.data = nil
  self.filenames = nil
  collectgarbage()
end
