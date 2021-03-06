![person1](./sample/img.png)

![person2](./sample/img2.png)

# BlazeFace_person_pytorch
Blazeface trained on pascal_voc person.

This repo is an **unofficial** implementation of:

`BlazeFace: Sub-millisecond Neural Face Detection on Mobile GPUs`
[Paper](https://arxiv.org/abs/1907.05047)

The repo contains..

- [x] SSD-like model
- [x] Training script
- [x] Eval script
- [x] Focal loss
- [ ] Trained weights
- [ ] CenterNet-like detection for more accuracy and faster opereations. (Under dev).

The training and inference is **> 10x faster** than SSD.

The BlazeFace model is based on [here](https://github.com/tkat0/PyTorch_BlazeFace), and localaization and detection layers are added from ssd.pytorch. 

Also the nms functions are borrowed from ssd.pytorch as well.

Thank you!

## requirements
```
Pytorch > 1.0
opencv
scikit-learn
```

## how to train
Run:

`python train_BlazeFace.py`

See train_BlazeFace.ipynb for specifics.

`Dataset_test_with_BlazeFace128-VOC-allclasses.ipynb` runs all VOC classes, but doesn't get good accuracy at this point.

## Inference
See `inference.ipynb` to run inference.

It takes less than 30ms on K80 to run single image (including nms).


