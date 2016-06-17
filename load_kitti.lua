local label_data = torch.load(param_data_dir .. 'label_colors.t7')
param_num_classes = 11

param_batch_size = 1
param_epoch_size = 1.0
--paths.dofile('data_container.lua')
local train_container = DataContainer {
  data_dir = param_data_dir,
  batch_size = param_batch_size,
  prefix = 'train',
  epoch_size = param_epoch_size,
}
local validation_container = DataContainer {
  data_dir = param_data_dir,
  batch_size = param_batch_size,
  prefix = 'valid',
  epoch_size = param_epoch_size,
}

param_class_names = label_data[1]
param_label_colors = label_data[2]

return train_container, validation_container
