# Kaggle: HPA Single Cell Classification Challenge

My top 6% approach to the [HPA Single Cell Classification Kaggle Competition](https://www.kaggle.com/c/hpa-single-cell-image-classification). The code includes peak response maps, puzzle-cam, and transformer encoders using RoI (cell mask) pooling over the CNN feature maps.

Overview of the cell-mask-based Transformer model (click on the image for higher resolution):

![cell transformer overview](https://github.com/martin-chobanyan/hpa-single-cell/blob/main/resources/cell-transformer-overview.png)

Overview of the cell-mask-based RoI pooling layer with cell position encoding:
![cell RoI pool overview](https://github.com/martin-chobanyan/hpa-single-cell/blob/main/resources/cell-roi-pool-overview.png)


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
