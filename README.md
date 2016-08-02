This is the code for the paper:

[Convolutional Scale Invariance for Semantic Segmentation](https://ivankreso.github.io/publications) - [GCPR 2016 (Oral)](http://www.kcmweb.de/conferences/gcpr2016)


We provide:
- Dataset preparation in `preprocessing/`
- Training code
- Implementation for data prefetching useful if the dataset can't fit into RAM


## Usage:
- Prepare the dataset for multi-scale or single-scale model.
- Run training script:
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
