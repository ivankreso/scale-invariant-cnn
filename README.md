This is the code for the paper:

[Convolutional Scale Invariance for Semantic Segmentation](https://ivankreso.github.io/publications) - [GCPR 2016 (Oral)](http://www.kcmweb.de/conferences/gcpr2016)


We provide:
- [Training code](https://github.com/ivankreso/scale-invariant-cnn/blob/master/train.lua)
- [Model definitions](https://github.com/ivankreso/scale-invariant-cnn/tree/master/models)
- Implementations of [weighted cross-entropy](https://github.com/ivankreso/scale-invariant-cnn/blob/master/layers/SpatialCrossEntropyCriterionWithIgnore.lua) loss funcion and [scale-selecion layer](https://github.com/ivankreso/scale-invariant-cnn/blob/master/layers/PyramidMultiplexer.lua)
- [Dataset preparation code](https://github.com/ivankreso/scale-invariant-cnn/tree/master/preprocessing) 
- Code for [multi-threaded data prefetching](https://github.com/ivankreso/scale-invariant-cnn/blob/master/data_container_multifile_threaded.lua) useful if the whole dataset can't fit into RAM


#### Usage:
- Download [KITTI](http://www.zemris.fer.hr/~ssegvic/multiclod/kitti_semseg_unizg.shtml) or [Cityscapes](https://www.cityscapes-dataset.com) dataset
- Prepare the data for [scale-invariant](https://github.com/ivankreso/scale-invariant-cnn/blob/master/models/scale_invariant.lua) or [single-scale](https://github.com/ivankreso/scale-invariant-cnn/blob/master/models/single_scale.lua) model
- Modify all data paths for your system and run the training script:
```
th train.lua -model models/scale_invariant.lua -solver solver\_config.lua
```


If you find this code useful in your research, please cite:

```
@inproceedings{kreso16gcpr,
  title={Convolutional Scale Invariance for Semantic Segmentation},
  author={Krešo, Ivan and Čaušević, Denis and Krapac, Josip and Šegvić, Siniša},
  booktitle={German Conference on Pattern Recognition},
  year={2016},
  organization={Springer}
}
```
