require 'cudnn'
require 'image'

img = image.load("/home/kivan/datasets/KITTI/segmentation/semantic_segmentation/union/train/data/rgb/00_000000.png"):cuda()
conv_sz = 3
conv = cudnn.SpatialConvolution(3, 64, conv_sz, conv_sz):cuda()
cout = conv:forward(img)
