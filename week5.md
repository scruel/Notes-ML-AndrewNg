[TOC]

# 9 神经网络: 学习(Neural Networks: Learning)

## 9.1 代价函数(Cost Function)

对于神经网络的代价函数公式：

$\begin{gather*} J(\Theta) = - \frac{1}{m} \sum_{i=1}^m \sum_{k=1}^K \left[y^{(i)}_k \log ((h_\Theta (x^{(i)}))_k) + (1 - y^{(i)}_k)\log (1 - (h_\Theta(x^{(i)}))_k)\right] + \frac{\lambda}{2m}\sum_{l=1}^{L-1} \sum_{i=1}^{s_l} \sum_{j=1}^{s_{l+1}} ( \Theta_{j,i}^{(l)})^2\end{gather*}$

> $L$: 神经网络的总层数
>
> $s_l$: 第 $l$ 层激活单元的数量（不包含偏置单元）
>
> $K$: 分类总数，即输出层输出单元的数量
>
> $h_\Theta(x)_k$: 分为第 $k$ 个分类的概率 $P(y=k | x ; \Theta) $
>
> 
>
> 注：此处符号表达和第四周的内容有异有同，暂时先按照视频来，有必要的话可以统一一下.

公式可长可长了是吧，那就对照下逻辑回归中的代价函数：

$J(\theta) = - \frac{1}{m} \sum_{i=1}^m [ y^{(i)}\ \log (h_\theta (x^{(i)})) + (1 - y^{(i)})\ \log (1 - h_\theta(x^{(i)}))] + \frac{\lambda}{2m}\sum_{j=1}^n \theta_j^2$

在神经网络的代价函数中，

- 左边的变化实际上是为了求解 $K$ 分类问题，即公式会对每个样本特征都运行 $K$ 次，并依次给出分为第 $k$ 类的概率。
- 右边的正则化项比较容易理解，每一层有多维矩阵 $\Theta^{(l)}\in \mathbb{R}^{(s_l + 1)\times s_{l+1}}$，从左到右看这个三次求和式 $\sum\limits_{l=1}^{L-1}\sum\limits_{i=1}^{s_l}\sum\limits_{j=1}^{s_{l+1}}$ ，就是对每一层间的多维参数矩阵 $\Theta^{(l)}$ ，依次平方后求取其除了偏置参数部分的和值，并循环累加即得结果。

> $\mathbb{R}^{m}$: 即 $m$ 维向量
>
> $\mathbb{R}^{m\times n}$: 即 $m \times n$ 维矩阵

可见，神经网络背后的思想是和逻辑回归一样的。



## 9.2 反向传播算法(Backpropagation Algorithm)

## 9.3 直观理解反向传播(Backpropagation Intuition)

## 9.4 实现注意点: 参数展开(Implementation Note: Unrolling Parameters)

## 9.5 Gradient Checking

## 9.6 Random Initialization

## 9.7 Putting It Together

## 9.8 自主驾驶(Autonomous Driving)