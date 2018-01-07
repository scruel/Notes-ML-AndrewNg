[TOC]

# 4 多变量线性回归(Linear Regression with Multiple Variables)

## 4.1 多特征(Multiple Features)

对于一个要度量的对象，一般来说会有不同维度的多个特征。比如之前的房屋价格预测例子中，除了房屋的面积大小，可能还有房屋的年限、房屋的层数等等其他特征：

![](image/20180107_234509.png)

这里由于特征不再只有一个，引入一些新的记号

> $n$: 特征的总数 
>
>  ${x}^{\left( i \right)}$: 代表特征矩阵中第 $i$ 行，也就是第 $i$ 个训练实例。
>
>  ${x}_{j}^{\left( i \right)}$: 代表特征矩阵中第 $i$ 行的第 $j$ 个特征，也就是第 $i$ 个训练实例的第 $j$ 个特征。

参照上图，则记号的举例有，${x}^{(2)}\text{=}\begin{bmatrix} 1416\\\ 3\\\ 2\\\ 40 \end{bmatrix}, {x}^{(2)}_{1} = 1416$

多变量假设函数 $h$ 表示为：$h_{\theta}\left( x \right)={\theta_{0}}+{\theta_{1}}{x_{1}}+{\theta_{2}}{x_{2}}+...+{\theta_{n}}{x_{n}}$

对于 $\theta_0$，和单特征中一样，我们将其看作基础数值。例如，房价的基础价格。

参数向量的维度为 $n+1$，在特征向量中添加 $x_{0}$ 后，其维度也变为 $n+1$， 则运用线性代数，可对 $h$ 简化。 

$h_\theta\left(x\right)=\begin{bmatrix}\theta_0\; \theta_1\; ... \;\theta_n \end{bmatrix}\begin{bmatrix}x_0 \newline x_1 \newline \vdots \newline x_n\end{bmatrix}= \theta^T x$

> $\theta^T$: $\theta$ 矩阵的转置
>
> $x_0$: 为了计算方便我们会假设 $x_0^{(i)} = 1$

## 4.2 Gradient Descent for Multiple Variables



## 4.3 Gradient Descent in Practice I - Feature Scaling

## 4.4 Gradient Descent in Practice II - Learning Rate

## 4.5 Features and Polynomial Regression

## 4.6 Normal Equation

## 4.7 Normal Equation Noninvertibility

## 4.8 Working on and Submitting Programming Assignments

# 5 Octave Matlab Tutorial

复习时可直接倍速回顾视频，笔记整理暂留。

## 5.1 Basic Operations

## 5.2 Moving Data Around

## 5.3 Computing on Data

## 5.4 Plotting Data

## 5.5 Control Statements_ for, while, if statement

## 5.6 Vectorization

## 5.x 常用函数整理