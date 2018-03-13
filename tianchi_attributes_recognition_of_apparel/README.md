# 天池服饰属性识别
* 大赛提供coat、collar、lapel、neck、neckline、pant、skirt、sleeve 共8类服饰图片，每一类服饰图片都有自己的属性
* 例：裤子有以下属性
  - Invisible
  - Short Pant
  - Mid Length
  - 3/4 Length
  - Cropped Pant
  - Full Length
* 参赛者需要预测给定的图片的属性，每张图片只有一个属性
* 分类模型基于keras 的ResNet50

* 2017-03-13 提交结果
 * score:0.9095 
 * basic-presicion:0.7566

## 项目工程结构

Before running the code, you should build up folders as follow:

```
├── models
|   |__ResNet50_pre_train_no_top.h5
|   |__ResNet50
|      |__resnet50_skirt_best.h5
|      |__resnet50_collar_best.h5
|      |__resnet50_lapel_best.h5
|      |__resnet50_neck_best.h5
|      |__resnet50_neckline_best.h5
|      |__resnet50_pant_best.h5
|      |__resnet50_sleeve_best.h5
|      |__resnet50_coat_best.h5
|
├── result
|   |__sub_0313a
|   |__sub_0313b
|   |__sub_0314a
|   |__... ...
|
├── test_a
|   |__coat_length_labels
|   |__collar_design_labels
|   |__lapel_design_labels
|   |__neck_design_labels
|   |__neckline_design_labels
|   |__pant_length_labels
|   |__skirt_length_labels
|   |__sleeve_length_labels
|
└── train
|   |__coat_length_labels
|   |__collar_design_labels
|   |__lapel_design_labels
|   |__neck_design_labels
|   |__neckline_design_labels
|   |__pant_length_labels
|   |__skirt_length_labels
|   |__sleeve_length_labels

```
## 各类别服饰属性
### skirt_length_labels

+ AttrKey : skirt_length_labels
+ AttrValues :
  - Invisible
  - Short Length
  - Knee Length
  - Midi Length
  - Ankle Length
  - Floor Length

### coat_length_labels

+ AttrKey : coat_length_labels
+ AttrValues :
  - Invisible
  - High Waist Length
  - Regular Length
  - Long Length
  - Micro Length
  - Knee Length
  - Midi Length
  - Ankle&Floor Length

### collar_design_labels

+ AttrKey : collar_design_labels
+ AttrValues :
  - Invisible
  - Shirt Collar
  - Peter Pan
  - Puritan Collar
  - Rib Collar

### lapel_design_labels

+ AttrKey : lapel_design_labels
+ AttrValues :
  - Invisible
  - Notched
  - Collarless
  - Shawl Collar
  - Plus Size Shawl

### neck_design_labels

+ AttrKey : neck_design_labels
+ AttrValues :
  - Invisible
  - Turtle Neck
  - Ruffle Semi-High Collar
  - Low Turtle Neck
  - Draped Collar

### neckline_design_labels

+ AttrKey : neckline_design_labels
+ AttrValues :
  - Invisible
  - Strapless Neck
  - Deep V Neckline
  - Straight Neck
  - V Neckline
  - Square Neckline
  - Off Shoulder
  - Round Neckline
  - Sweat Heart Neck
  - One	Shoulder Neckline

### pant_length_labels

+ AttrKey : pant_length_labels
+ AttrValues :
  - Invisible
  - Short Pant
  - Mid Length
  - 3/4 Length
  - Cropped Pant
  - Full Length

### sleeve_length_labels

+ AttrKey : sleeve_length_labels
+ AttrValues :
  - Invisible
  - Sleeveless
  - Cup Sleeves
  - Short Sleeves
  - Elbow Sleeves
  - 3/4 Sleeves
  - Wrist Length
  - Long Sleeves
  - Extra Long Sleeves





