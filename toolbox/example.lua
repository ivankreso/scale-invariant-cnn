require('torch')
require('nn')
require('cudnn')
require 'nngraph'
Sanitize = require('sanitize')


-- define model
--local model = nn.Sequential()
----model:add(nn.SpatialConvolutionMM(3, 16, 5, 5))
--model:add(cudnn.SpatialConvolution(3, 16, 5, 5))
--model:add(nn.SpatialBatchNormalization(16))
----model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
--model:add(cudnn.SpatialMaxPooling(2, 2))


local input = nn.Identity()()
local conv = cudnn.SpatialConvolution(3, 16, 5, 5)(input)
local norm = nn.SpatialBatchNormalization(16)(conv)
local pool = cudnn.SpatialMaxPooling(2, 2)(norm)

model = nn.gModule({input}, {pool})
model:cuda()

model1 = model:clone()
w = model:getParameters()
w1 = model1:getParameters()
print((w - w1):sum())

-- set input
x = torch.Tensor(128, 3, 32, 32):cuda()

-- compute dummy forward-backward
--y = model:forward(x)
--dx = model:backward(x, y)
-- save model
torch.save('model.t7', model)
torch.save('model-cleaned.t7', Sanitize(model))
-- chk filesize
os.execute('du -sh model.t7')
os.execute('du -sh model-cleaned.t7')
-- test cleaned model if it still works
local model_cleaned  = torch.load('model-cleaned.t7')
model_cleaned:forward(x)
-- remove temp files
os.execute('rm -f model.t7')
os.execute('rm -f model-cleaned.t7')
