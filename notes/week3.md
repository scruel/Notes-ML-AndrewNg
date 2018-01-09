[TOC]

# 6 逻辑回归(Logistic Regression)

## 6.1 分类(Classification)

在分类问题中，预测的结果是离散值（结果是否属于某一类），逻辑回归算法(Logistic Regression)被用于解决这类分类问题。

- 垃圾邮件判断
- 金融欺诈判断
- 肿瘤诊断

肿瘤诊断问题：

![](image/20180109_144040.png)

肿瘤诊断问题是一个二元分类问题，则定义 $ y \in\lbrace 0, 1\rbrace$，其中 0 表示**负向类(negative class)**，代表恶性肿瘤，1 为**正向类(positive class)**，代表良性肿瘤。如图，定义最右边的样本为**偏差项**。

在未加入偏差项时，线性回归算法给出了品红色的拟合直线，若规定

$h_\theta(x) \geqslant 0.5$ ，预测为 $y = 1$，即正向类；

$h_\theta(x) \lt 0.5$ ，预测为 $y = 0$，即负向类。

即以 0.5 为分类**阈值**(threshold)，则我们就可以根据线性回归结果，得到相对正确的分类结果 $y$。



接下来加入偏差项，线性回归算法给出了靛青色的拟合直线，如果阈值仍然为 0.5，对于明明属于负向类的情况，算法会给出正向类这个完全错误的结果。

不仅如此，线性回归算法的值域为 $R$，则当线性回归算法给出如 $h = 1000, h = -1000$ 等数值时，我们仍会给出结果 $y \in \lbrace 0, 1\rbrace$，这显得非常怪异。



区别于线性回归算法，逻辑回归算法是一个分类算法，**其输出值永远在 0 到 1 之间**，即 $h \in (0,1)$。

## 6.2 假设函数表示(Hypothesis Representation)



## 6.3 Decision Boundary

## 6.4 Cost Function

## 6.5 Simplified Cost Function and Gradient Descent

## 6.6 Advanced Optimization

## 6.7 Multiclass Classification_ One-vs-all

# 7 Regularization
## 7.1 The Problem of Overfitting

## 7.2 Cost Function

## 7.3 Regularized Linear Regression

## 7.4 Regularized Logistic Regression