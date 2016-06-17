
p = torch.load('/mnt/ikreso/datasets/Cityscapes/pyramid/1504x672_8s/results/val/pyramid_val_iou_per_image.t7')
b = torch.load('/mnt/ikreso/datasets/Cityscapes/results/baseline/val/baseline_val_iou_per_image.t7')
p_iou = p[1]
b_iou = b[1]

diff = torch.DoubleTensor(#p[2])
for i = 1, #p[2] do
  for j = 1, #b[2] do
    if p[2][i] == b[2][j] then
      diff[i] = p_iou[i] - b_iou[j]
    end
  end
end
--print(diff)
--diff, indices = torch.sort(diff, 1, true)
--diff, indices = torch.topk(diff, 10, 1, true, true)
diff, indices = torch.topk(p_iou, 10, 1, false, true)
print(diff)
for i = 1, indices:size(1) do
  print(p[2][indices[i]])
end

--imgs = {'lindau_000001_000019.png', 'frankfurt_000001_029086.png', 'frankfurt_000001_014565.png',
--'frankfurt_000000_010351.png', 'frankfurt_000001_010600.png', 'munster_000041_000019.png',
--'munster_000035_000019.png', 'frankfurt_000001_050686.png', 'frankfurt_000001_035864.png',
--'frankfurt_000001_058057.png'}

--imgs = {'munster_000128_000019.png'}
--local img_pyr = '/mnt/ikreso/datasets/Cityscapes/pyramid/1504x672_8s/results/val/img/'
--local img_bas = '/mnt/ikreso/datasets/Cityscapes/results/baseline/val/img/'
----local img_raw_dir = '/mnt/ikreso/datasets/Cityscapes/1504x672/img/data/val/'
----local save_dir_raw = '/mnt/ikreso/datasets/Cityscapes/results/comp/raw/'
--local save_dir_pyr = '/mnt/ikreso/datasets/Cityscapes/results/comp/pyramid/'
--local save_dir_bas = '/mnt/ikreso/datasets/Cityscapes/results/comp/baseline/'
----
----os.execute('mkdir -p ' .. save_dir_raw)
----os.execute('mkdir -p ' .. save_dir_pyr)
----os.execute('mkdir -p ' .. save_dir_bas)
----
--for i = 1, #imgs do
--  os.execute('cp ' .. img_pyr .. imgs[i] .. ' ' .. save_dir_pyr .. imgs[i])
--  os.execute('cp ' .. img_bas .. imgs[i] .. ' ' .. save_dir_bas .. imgs[i])
--  --os.execute('cp ' .. img_raw_dir .. imgs[i] .. ' ' .. save_dir_raw .. imgs[i])
--end
