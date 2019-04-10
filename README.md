# 亿保杯--票据分割比赛

---
<Excerpt in index | 首页摘要> 
 **<font color = #80000>这只是浙大的一个校内比赛**
**KeyWords Plus**:     &#160;&#160;&#160;&#160;     	关于票据重叠文本的情况进行分割
- **relevant** ：出发点：很多票据会出现打印字错位，甚至是打印字和底板重叠，利用分割算法进行分割重叠文本，以便于后期的单据识别。
- **coding** ：[Github](https://github.com/weijiawu/Yibao-cup_competition)


## **简介**
 &#160;&#160;&#160;&#160;  &#160;&#160;&#160;&#160; **初赛十分简单，就是用分割网络对重叠文本进行分割即可，评价指标是IOU**

![Alt text](https://github.com/weijiawu/Yibao-cup_competition/tree/master/picture/1554865527664.png)


**<font size = 5><font color=#000555555>数据集如下所示：**

![Alt text](https://github.com/weijiawu/Yibao-cup_competition/tree/master/picture/1554865681406.png)

**<font size = 5><font color=#000555555>label：**

<div align=center>![Alt text](https://github.com/weijiawu/Yibao-cup_competition/tree/master/picture/1554865755538.png)

`目的就是将两个重叠数字分割开来，从实际场景出发就分割开收据上的重叠文本，以便于后期的信息采集处理。`

## **我们的解决方案**

 &#160;&#160;&#160;&#160;   &#160;&#160;&#160;&#160;  **因比赛提供的人工合成数据集和实际场景中的重叠文本差别太大，因此我们的方案也分为两部分：**


**<font size=4.5>1. 基于Deeplab网络的文本分割算法**
 &#160;&#160;&#160;&#160;   &#160;&#160;&#160;&#160; &#160;&#160;&#160;&#160;   &#160;&#160;&#160;&#160; &#160;&#160;&#160;&#160;   &#160;&#160;&#160;&#160; &#160;&#160;&#160;&#160;   &#160;&#160;&#160;&#160; &#160;&#160;&#160;&#160;   &#160;&#160;&#160;&#160; &#160;&#160;&#160;&#160;   &#160;&#160;&#160;&#160; &#160;&#160;&#160;&#160;   &#160;&#160;&#160;&#160; &#160;&#160;&#160;&#160;   &#160;&#160;&#160;&#160;--------`主要针对初赛分割任务`

**<font size=4.5>2. 基于弱监督学习的分割识别网络**
 &#160;&#160;&#160;&#160;   &#160;&#160;&#160;&#160; &#160;&#160;&#160;&#160;   &#160;&#160;&#160;&#160;  &#160;&#160;&#160;&#160;   &#160;&#160;&#160;&#160; &#160;&#160;&#160;&#160;   &#160;&#160;&#160;&#160; &#160;&#160;&#160;&#160;   &#160;&#160;&#160;&#160; &#160;&#160;&#160;&#160;   &#160;&#160;&#160;&#160; &#160;&#160;&#160;&#160;   &#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;   &#160;&#160;&#160;&#160;--------`从实际场景出发解决单据文本重叠方案`


### **方案一_针对初赛的分割任务**

在初赛我们主要尝试了三种网络框架：

 &#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;（1）、具有膨胀卷积的**Deeplab**（线上0.97的指标）
 
 &#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;（2）、参考了TextSnake的**FCN**（线上0.955的指标）
 
 &#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;（3）、基于VGG11的**U-net**（线上0.965的指标）


<font size=5><font color=#00888000>**Deeplab_v3**

 &#160;&#160;&#160;&#160;Deeplab网络我就不具体介绍了，网上有很多资料，在分割领域也十分有名

他主要的特殊就是引入了膨胀卷积，提出了**DRN（Dilated Residual Networks）**，他与Resnet，VGG主要的区别如下图

<div align=center> ![Alt text](https://github.com/weijiawu/Yibao-cup_competition/tree/master/picture/1554866665604.png)

<font size=5><font color=#00888000>**FCN**

 &#160;&#160;&#160;&#160;**FCN（Fully Convolutional Network)的一个改进版本作为基本分割网络框架**，其实就是TextSnake的网络，基础结构为vgg16，取出下采样中的不同的五层特征进行上采样融合，主要加入的元素有FPN（Feature Pyramid Networks。
<div align=center>![Alt text](https://github.com/weijiawu/Yibao-cup_competition/tree/master/picture/1554875089380.png)

<font size=5><font color=#00888000>**FCN**

 &#160;&#160;&#160;&#160;**U-net**用的是一个kaggle汽车分割冠军方案的网络，基础结构为vgg11。**传送门：**[Github](https://github.com/asanakoy/kaggle_carvana_segmentation)

 <div align=center> ![Alt text](https://github.com/weijiawu/Yibao-cup_competition/tree/master/picture/1554875437472.png)


<font size=5><font color=#00888000>**Loss:**

**用分水岭算法，可以对边缘像素进行加权。**
 &#160;&#160;&#160;&#160; &#160;&#160;&#160;&#160;主要是计算 BCE loss 时，mask数字边缘 的像素的权重是在文本数字里面像素的2 倍，而重合部分的像素点是不重合部分像素的2倍.因为考虑到训练到后期，文本内部的像素点的置信度都是很高的，只有在边缘的像素点才有可能被预测错误，而重合部分权重加大是因为分割重叠的数字是本次分割任务的主要目的，并且非重叠区域往往可以分割的较好。
 
 &#160;&#160;&#160;&#160; &#160;&#160;&#160;&#160;总的Loss函数设计思路：
**<div align=center><font color=#0088888>f(x) = BCE + 1 - DICE.**

<div align=center>![Alt text](https://github.com/weijiawu/Yibao-cup_competition/tree/master/picture/1554875794374.png)


### **方案二_针对现实场景下的弱监督识别网络**

 &#160;&#160;&#160;&#160; &#160;&#160;&#160;&#160;刚开始我也尝试用**分割**去分割开打印上去的字体与底板，**发现有如下难点：**
&#160;&#160;&#160;&#160; 1、`现实场景中文本过于精细复杂难以分割。`
（每个字体的笔画只有2、3像素宽，且干扰严重，这对分割网络要求太高，是很难进行精细分割。）

&#160;&#160;&#160;&#160; 2、`现实场景中的分割label难以获取。`
（大多数分割任务的对象都是一个较大的实体对象，而单据中的文字级别的分割label过于精细，基本不可能根据单据上进行标出）

&#160;&#160;&#160;&#160; 3、`现实中的单据形状尺寸相差很大。`
其次精细分割任务中忌讳使用resize，而现实中单据形状大小都不一致，甚至可以说差距很大，因此大多数分割网络在单据分割中鲁棒性不强。

<div align=center> ![Alt text](https://github.com/weijiawu/Yibao-cup_competition/tree/master/picture/1554876031923.png)


**<font color =#008888888><font size=5>我们提出的弱监督分割识别算法到底是做什么事情呢？**

&#160;&#160;&#160;&#160; 弱监督学习分割识别网络 ———— **<font size=5>可以识别重叠文本**

<div align=center>![Alt text](https://github.com/weijiawu/Yibao-cup_competition/tree/master/picture/1554876347227.png)


具体算法思路我就不在这里展开了，暂时不能开源。下面是一些人工合成的重叠单词数据集。

&#160;&#160;&#160;&#160; &#160;&#160;&#160;&#160; **数据生成说明：**数据底板采样于收据，单词收集于学术论文，颜色随机，**大小仿照真实场合文本大小，**在200*32的图片上随机重合生成（3万训练集，5千测试集）


<div align=center> ![Alt text](https://github.com/weijiawu/Yibao-cup_competition/tree/master/picture/1554876582009.png)


&#160;&#160;&#160;&#160; &#160;&#160;&#160;&#160; 在该生成数据上，**弱监督分割识别网络baseline在可以达到0.71**的准确率（备注：因时间有限，只用了一天的时间进行的匆忙实验）


&#160;&#160;&#160;&#160; &#160;&#160;&#160;&#160;<font size=5>$$准确率 = 预测正确的单词数量/测试集单词总数 $$

![Alt text](https://github.com/weijiawu/Yibao-cup_competition/tree/master/picture/1554876563891.png)

