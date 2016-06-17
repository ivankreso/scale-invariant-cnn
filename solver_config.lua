param_home_dir = '/home/kivan/'
param_vgg_prototxt = "/home/kivan/datasets/pretrained/caffe/vgg_16/VGG_ILSVRC_16_layers_deploy.prototxt.txt"
param_vgg_model = "/home/kivan/datasets/pretrained/caffe/vgg_16/VGG_ILSVRC_16_layers.caffemodel"


paths.dofile('solver_config_cityscapes.lua')







--param_optimization = 'sgd'
--param_optim_conf = {
--  -- best so far:
--  --learningRate = 1e-7,
--  learningRate = 1e-5,
--  --momentum = 0.99,
--  weightDecay = 0.0005,
--  momentum = 0.9
--
--  --learningRate = 1e-6,
--  --learningRate = 1e-8,
--
--  --learningRate = 1e-6,
--  --momentum = 0.9,
--
--  --learningRate = 1e-5,
--  --learningRate = 1e-9,
--  --momentum = 0.99,
--  --momentum = 0.7,
--  --learningRateDecay = 1e-5
--}

--param_optimization = 'rmsprop'
--param_optim_conf = {
--  -- best
--  learningRate = 1e-3
--  --learningRate = 1e-4
--}

--param_optimization = 'adadelta'
--param_optim_conf = {}

--param_optimization = 'adagrad'
--param_optim_conf = {
--  --learningRate = 1e-3,
--  learningRate = 1e-7,
--  --learningRateDecay = 1e-5
--}

---- 0.001 - 0.1
--param_lr = 0.1
---- 1e-7 - 1e-6
--param_momentum = 0.9
----param_weight_decay = 1e-4
--param_weight_decay = 1e-4
----0.01 / (1 + 10*(3210000/200)*1e-4)
----param_lr_decay = 1e-6
--param_lr_decay = 1e-5

--param_lr_decay = 1e-7
--param_momentum = 0.5
--param_lr = 0.01
--param_weight_decay = 1e-5

---- ASGD:
--param_optimization = 'asgd'
--optimState = {
--  -- lr
--  eta0 = 1.0,
--  -- decay
--  lambda = 5e-4,
--  alpha = 0.9
--}
