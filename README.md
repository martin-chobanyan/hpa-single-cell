# Kaggle: HPA Single Cell Classification Challenge

My approach to the [HPA Single Cell Classification Kaggle Competition](https://www.kaggle.com/c/hpa-single-cell-image-classification). The code includes peak response maps, puzzle-cam, and transformer encoders using RoI (cell mask) pooling over the CNN feature maps.

### Installation
To install the baseline cell segmentation model, run the following:
```
pip install https://github.com/CellProfiling/HPA-Cell-Segmentation/archive/master.zip
```

### Execution Order
- download public dataset
- exclude images
- multilabel stratify split
- train
