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
* 公开了部分源码，其他部分赛后开源

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
|
└── train.csv
|
└── test_a.csv
|
└── resnet_fine_tune.ipynb

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

## 评估指标

*  录入参赛者提交的csv文件，为每条数据计算出AttrValueProbs中的最大概率以及对应的标签，分别记为MaxAttrValueProb和MaxAttrValue。
*  对每个属性维度，分别初始化评测计数器：
   * BLOCK_COUNT = 0 (不输出的个数)
   * PRED_COUNT = 0 (预测输出的个数)
   * PRED_CORRECT_COUNT = 0 (预测正确的个数)
   * 设定GT_COUNT为该属性维度下所有相关数据的总条数
*  给定一个模型输出阈值（ProbThreshold），分析与该属性维度相关的每条数据的预测结果：
   * 当MaxAttrValueProb < ProbThreshold，模型不输出：BLOCK_COUNT++
   * 当MaxAttrValueProb >= ProbThreshold：
   * MaxAttrValue对应的标注位是'y'时，记为正确： PRED_COUNT++，PRED_CORRECT_COUNT++
   * MaxAttrValue对应的标注位是'm'时，不记入准确率评测：无操作
   * MaxAttrValue对应的标注位是'n'时，记为错误： PRED_COUNT++
*  遍历使BLOCK_COUNT落在[0, GT_COUNT)里所有可能的阈值ProbThreshold，分别计算：
   * 准确率(P)：PRED_CORRECT_COUNT / PRED_COUNT
   * 统计它们的平均值，记为AP。 
*  综合所有的属性维度计算得到的AP，统计它们的平均值，得出mAP。mAP将作为挑战赛——服饰属性标签识别赛道的竞赛排名得分。

*  我们还会展示BasicPrecision指标，即模型在测试集全部预测输出(ProbThreshold=0)情况下每个属性维度准确率的平均值，作为更直接的准确率预估指标供大家参考。在BasicPrecision = 0.7时，排名得分mAP一般在 0.93 左右。





