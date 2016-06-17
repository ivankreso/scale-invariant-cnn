function Round(num, num_decimals)
  num_decimals = num_decimals or 1
  local fac = num_decimals * 10
  return torch.round(num * fac) / fac
end

function PrintAndSave(str)
  if param_no_print == nil then
    print(str)
  end
  if error_logfile ~= nil then
    error_logfile:writeString(str .. '\n')
    error_logfile:synchronize()
  end
end

function PrintEvaluationStats(confusion_matrix, name)
  local num_correct = confusion_matrix:trace()
  local num_classes = confusion_matrix:size(1)
  local total_size = confusion_matrix:sum()
  local avg_pixel_acc = num_correct / total_size * 100.0
  local TPFN = confusion_matrix:sum(1):view(num_classes)
  local TPFP = confusion_matrix:sum(2):view(num_classes)
  local FN = TPFN - confusion_matrix:diag()
  local FP = TPFP - confusion_matrix:diag()
  --local class_acc = torch.FloatTensor(size_per_class:size(1))
  local class_acc = torch.FloatTensor(num_classes)
  local class_recall = torch.FloatTensor(num_classes)
  local class_precision = torch.FloatTensor(num_classes)
  PrintAndSave('\n----- ' .. name .. ' error report -----')
  local latex_str = ''
  for i = 1, num_classes do
    --class_acc[i] = confusion_matrix[{i,i}] / size_per_class[i] * 100.0
    --class_acc[i] = confusion_matrix[{i,i}] / size_per_class[i] * 100.0
    local TP = confusion_matrix[{i,i}]
    if TP > 0 then
      class_acc[i] = (TP / (TP + FP[i] + FN[i])) * 100.0
    else
      class_acc[i] = 100
    end
    if TPFN[i] > 0 then
      class_recall[i] = (TP / TPFN[i]) * 100.0
    else
      class_recall[i] = 100
    end
    if TPFP[i] > 0 then
      class_precision[i] = (TP / TPFP[i]) * 100.0
    else
      class_precision[i] = 100
    end
    local class_name
    if param_dataset_name == 'cityscapes' then
      class_name = param_label_colors[i][1]
    else
      class_name = param_label_colors[i][4]
    end
    local str = string.format("\t%s IoU accuracy = %.2f %%", class_name, class_acc[i])
    latex_str = latex_str .. ' & ' .. string.format('%0.1f', Round(class_acc[i]))
    PrintAndSave(str)
  end
  local str = name .. string.format(" pixel accuracy = %.2f %%", avg_pixel_acc)
  PrintAndSave(str)
  local avg_class_acc = class_acc:mean()
  PrintAndSave(latex_str .. '& ' .. Round(avg_class_acc))
  str = name .. string.format(" IoU mean class accuracy - TP / (TP+FN+FP) = %.2f %%", avg_class_acc)
  PrintAndSave(str)
  local avg_class_recall = class_recall:mean()
  str = name .. string.format(" mean class recall - TP / (TP+FN) = %.2f %%", avg_class_recall)
  PrintAndSave(str)
  local avg_class_precision = class_precision:mean()
  str = name .. string.format(" mean class precision - TP / (TP+FP) = %.2f %%", avg_class_precision)
  PrintAndSave(str)
  return avg_pixel_acc, avg_class_acc, avg_class_recall, avg_class_precision, total_size
end


function AggregateStatsWithDepth(y_mat, yt_mat, confusion_matrix, depth_img,
                                 disp_correct_cnt, disp_total_cnt)
  local rows = y_mat:size(1)
  local cols = y_mat:size(2)
  local nclasses = confusion_matrix:size(2)
  local yt = torch.data(yt_mat:contiguous())
  local y = torch.data(y_mat:contiguous())
  local c_mat = torch.data(confusion_matrix:contiguous())

  local depth_ptr = torch.data(depth_img:contiguous())
  local max_disp = disp_correct_cnt:size(2)
  local disp_correct_ptr = torch.data(disp_correct_cnt:contiguous())
  local disp_total_ptr = torch.data(disp_total_cnt:contiguous())
  for r = 0, rows-1 do
    local stride = r * cols
    for c = 0, cols-1 do
      local lt = yt[stride + c]
      if lt > 0 then
        lt = lt - 1
        local l = y[stride + c] - 1
        c_mat[l*nclasses + lt] = c_mat[l*nclasses + lt] + 1
        local disp = depth_ptr[stride + c]
        if disp < max_disp then
          local pos = l * max_disp + disp
          disp_total_ptr[pos] = disp_total_ptr[pos] + 1
          --print(l .. ' == ' .. lt)
          if lt == l then
            --print(r, c, l, disp)
            disp_correct_ptr[pos] = disp_correct_ptr[pos] + 1
          end
        end
      end
    end
  end
end

function AggregateStats(y_mat, yt_mat, confusion_matrix)
  local rows = y_mat:size(1)
  local cols = y_mat:size(2)
  local nclasses = confusion_matrix:size(2)
  local yt = torch.data(yt_mat:contiguous())
  local y = torch.data(y_mat:contiguous())
  local c_mat = torch.data(confusion_matrix:contiguous())

  for r = 0, rows-1 do
    local stride = r * cols
    for c = 0, cols-1 do
      local lt = yt[stride + c]
      if lt > 0 then
        lt = lt -1
        local l = y[stride + c] - 1
        local idx = l*nclasses + lt
        c_mat[idx] = c_mat[idx] + 1
      end
    end
  end
end

function WriteTensor(tensor, filename)
  if not tensor then return false end
  local file = torch.DiskFile(filename, "w"):binary()
  --print("Tensor size = ", tensor:nElement())
  local size = tensor:size()
  --print(size:size())
  file:writeInt(size:size())
  file:writeLong(size)
  local data = tensor:storage()
  assert(file:writeFloat(data) == tensor:nElement())
  file:close()
end


function DrawPrediction(y)
  local h = y:size(1)
  local w = y:size(2)
  local img = torch.ByteTensor(3, h, w):fill(0)
  local cimg = torch.data(img:contiguous())
  local cy = torch.data(y:int():contiguous())
  local s1 = h * w
  for i = 0, h-1 do
    local s2 = i * w
    for j = 0, w-1 do
      --local l = y[{i,j}]
      local l = cy[s2 + j]
      --print(l)
      local color = param_label_colors[l][2]
      for c = 0, 2 do
        --img[{c,i,j}] = color[c]
        --cimg[c*s1 + s2 + j] = color[c+1]
        cimg[c*s1 + s2 + j] = color[c+1]
      end
    end
  end
  --image.save(img_save_dir .. img_num, img)
  return img
end


function PyramidFreeForward(net, x)
  local curr_out = {}
  for i = 1, #x-1 do
    local one_net = net:get(i)
    local out = one_net:forward(x[i])
    --print(out:size())
    --if type(out) == 'table' then
    --  local tmp = {}
    --  for j = 1, #out do
    --    table.insert(tmp, out[j]:clone())
    --  end
    --  table.insert(curr_out, tmp)
    --else
    --  table.insert(curr_out, out:clone())
    --end
    table.insert(curr_out, out:clone())
    one_net:clearState()
  end
  table.insert(curr_out, x[#x])

  for i = #x+1, net:size() do
    local next_out = net:get(i):updateOutput(curr_out)
    curr_out = next_out
  end
  return curr_out
end

--function AggregateConfusionMatrix(y, yt, confusion_matrix)
--  local rows = y:size(1)
--  local cols = y:size(2)
--  for r = 1, rows do
--    for c = 1, cols do
--      local lt = yt[{r,c}]
--      if lt > 0 then
--        local l = y[{r,c}]
--        confusion_matrix[{lt,l}] = confusion_matrix[{lt,l}] + 1
--      end
--    end
--  end
--end
