param_max_epoch = 40
--param_max_epoch = 5
param_training = true
--param_iters_between_val = 1500
param_iters_between_val = 5000
param_optimization = 'adam'
param_optim_conf = {
  -- best for 1. epoch
  --learningRate = 1e-5
  --learningRate = 1e-4
}
--param_lr_policy = {1e-4, 1e-5, 1e-5, 5e-6}
--param_lr_policy = {1e-4, 5e-5, 1e-5, 1e-5, 1e-5, 5e-6}
param_lr_policy = {}
param_lr_policy[1] = 1e-5
--param_lr_policy[7] = 5e-6
--param_lr_policy[31] = 1e-6
-- epoch is 1/2:
--param_lr_policy[5] = 5e-6
--param_lr_policy[10] = 1e-6
--param_lr_policy[21] = 1e-6

param_lr_policy[3] = 5e-6
param_lr_policy[6] = 1e-6
--param_lr_policy[12] = 5e-7
