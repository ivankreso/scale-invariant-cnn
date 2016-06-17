
--paths.dofile('data_container_multifile.lua')
paths.dofile('data_container_multifile_threaded.lua')
paths.dofile('data_container_multifile_threaded_val.lua')
param_dataset_name = 'cityscapes'
param_label_colors, param_class_names, param_num_classes =
  paths.dofile('preprocessing/cityscapes/class_colors.lua')

local train_container = nil
if not param_skip_train_data then
  train_container = DataContainerMultiFileThreaded {
    data_dir = param_data_dir,
    prefix = 'train',
    num_files = param_num_train_files or 60,
    --num_files = param_num_train_files or 30,
    --num_files = param_num_train_files or 5,
    --num_files = 2,
    iters_between_val = param_iters_between_val
    --iters_between_val = 2
  }
end

local validation_container = nil
if not param_skip_val_data then
  validation_container = DataContainerMultiFileThreadedVal {
    data_dir = param_data_dir,
    prefix = 'val',
    num_files = param_num_valid_files or 10
    --num_files = param_num_valid_files or 5
  }
end

return train_container, validation_container
