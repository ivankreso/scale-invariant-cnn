
cam0 = {718.856, 718.856, 607.1928, 185.2157, 0.53716572}
cam3 = {721.5377, 721.5377, 609.5593, 172.854, 0.53715059}
cam4 = {707.0912, 707.0912, 601.8873, 183.1104, 0.53715065}

calib_table = {}
calib_table['00'] = cam0
calib_table['01'] = cam0
calib_table['02'] = cam0
calib_table['03'] = cam3
calib_table['04'] = cam4
calib_table['05'] = cam4
calib_table['06'] = cam4
calib_table['07'] = cam4
calib_table['08'] = cam4
calib_table['09'] = cam4
calib_table['10'] = cam4

local function Reconstruct3D(depth, filename)
  --print(depth)
  local seq = filename:sub(1,2)
  local cam_params = calib_table[seq]
  local f = cam_params[1]
  local cx = cam_params[3]
  local cy = cam_params[4]
  local b = cam_params[5]

  local height = depth:size(1)
  local width = depth:size(2)
  local depth_ptr = torch.data(depth:contiguous())
  local point_cloud = torch.FloatTensor(height, width, 3)
  local pc_ptr = torch.data(point_cloud:contiguous())
  for y = 0, height-1 do
    local stride1 = y * width
    local stride2 = y * width * 3
    for x = 0, width-1 do
      local d = math.max(depth_ptr[stride1 + x], 0.001)
      local pt_idx = stride2 + (x * 3)
      pc_ptr[pt_idx] = (x - cx) * b / d
      pc_ptr[pt_idx+1] = (y - cy) * b / d
      pc_ptr[pt_idx+2] = f * b / d
      --print(point_cloud[y+1][x+1])
    end
  end
  point_cloud = point_cloud:transpose(2,3)
  point_cloud = point_cloud:transpose(1,2)
  return point_cloud:contiguous()
end

local function Generate3DPoints(data_path, name)
  local depth_path = data_path .. name .. '_depth.t7'
  local disp_imgs = torch.load(depth_path)
  local filenames = torch.load(data_path .. name .. "_filenames.t7")
  local data_3d = {}
  local data_3d_norm = {}
  for i = 1, #disp_imgs do
    xlua.progress(i, #disp_imgs)
    --print(filenames[i])
    local point_cloud = Reconstruct3D(disp_imgs[i][2], filenames[i])
    table.insert(data_3d, point_cloud)
    local norm_point_cloud = (point_cloud - point_cloud:mean()) / point_cloud:std()
    table.insert(data_3d_norm, norm_point_cloud)
    --print(norm_point_cloud)
  end
  torch.save(data_path .. name .. '_data_3d_meters.t7', data_3d)
  torch.save(data_path .. name .. '_data_3d_normalized.t7', data_3d_norm)
end

local data_path = '/home/kivan/datasets/KITTI/segmentation/semantic_segmentation/union/torch/1216x352/'

Generate3DPoints(data_path, 'train')
Generate3DPoints(data_path, 'valid')
