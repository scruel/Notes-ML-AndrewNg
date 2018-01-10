[TOC]

# 6 逻辑回归(Logistic Regression)

## 6.1 分类(Classification)

在分类问题中，预测的结果是离散值（结果是否属于某一类），逻辑回归算法(Logistic Regression)被用于解决这类分类问题。

- 垃圾邮件判断
- 金融欺诈判断
- 肿瘤诊断

肿瘤诊断问题：

![](image/20180109_144040.png)

肿瘤诊断问题是一个**二元分类问题(binary class problems)**，则定义 $ y \in\lbrace 0, 1\rbrace$，其中 0 表示**负向类(negative class)**，代表恶性肿瘤("-")，1 为**正向类(positive class)**，代表良性肿瘤("+")。如图，定义最右边的样本为**偏差项**。

在未加入偏差项时，线性回归算法给出了品红色的拟合直线，若规定

$h_\theta(x) \geqslant 0.5$ ，预测为 $y = 1$，即正向类；

$h_\theta(x) \lt 0.5$ ，预测为 $y = 0$，即负向类。

即以 0.5 为**阈值**(threshold)，则我们就可以根据线性回归结果，得到相对正确的分类结果 $y$。



接下来加入偏差项，线性回归算法给出了靛青色的拟合直线，如果阈值仍然为 0.5，可以看到算法在某些情况下会给出完全错误的结果。

不仅如此，线性回归算法的值域为 $R$，则当线性回归函数给出诸如 $h = 10000, h = -10000$ 等很大/很小(负数)的数值时，结果 $y \in \lbrace 0, 1\rbrace$，这显得非常怪异。



区别于线性回归算法，逻辑回归算法是一个分类算法，**其输出值永远在 0 到 1 之间**，即 $h \in (0,1)$。

## 6.2 假设函数表示(Hypothesis Representation)

为了使 $h \in \left(0, 1\right)$，引入逻辑回归模型，定义假设函数
$$
h_\theta \left( x \right)=g(z)=g\left(\theta^{T}X \right)
$$
对比线性回归函数 $h_\theta \left( x \right)=\theta^{T}X$，$g$ 表示逻辑函数([logistic function][1])，复合起来，则称为逻辑回归函数。

逻辑函数是 S 形函数，会将所有实数映射到 $(0, 1)$ 范围。

[sigmoid 函数][2]（如下图）是逻辑函数的特殊情况，其公式为 $g\left( z \right)=\frac{1}{1+{{e}^{-z}}}$。 

![sigmoid function](image/2413fbec8ff9fa1f19aaf78265b8a33b_Logistic_function.png)

应用 sigmoid 函数，则逻辑回归模型：$$h_{\theta}(x)=g(\theta^Tx) =\frac{1}{1+e^{-\theta^Tx}}$$

逻辑回归模型中，$h_\theta \left( x \right)$ 的作用是，根据输入 $x$ 以及参数 $\theta$，计算得出”输出 $y=1$“的可能性(estimated probability)，概率学中表示为：

$\begin{align*}& h_\theta(x) = P(y=1 | x ; \theta) = 1 - P(y=0 | x ; \theta) \newline & P(y = 0 | x;\theta) + P(y = 1 | x ; \theta) = 1\end{align*}$



以肿瘤诊断为例，$h_\theta \left( x \right)=0.7$ 表示病人有 $70\%$ 的概率得了恶性肿瘤。

[1]: https://en.wikipedia.org/wiki/Logistic_function
[2]: https://en.wikipedia.org/wiki/Sigmoid_function

## 6.3 决策边界(Decision Boundary)

决策边界的概念可以帮助我们更好地理解逻辑回归模型。

在逻辑回归中，有假设函数 $h_\theta \left( x \right)=g(z)=g\left(\theta^{T}X \right)$。

为了得出分类的结果，这里和前面一样，规定以 $0.5$ 为阈值：


$\begin{align*}& h_\theta(x) \geq 0.5 \rightarrow y = 1 \newline& h_\theta(x) < 0.5 \rightarrow y = 0 \newline\end{align*}$

回忆一下逻辑函数的图像：

![sigmoid function](image/2413fbec8ff9fa1f19aaf78265b8a33b_Logistic_function.png)

观察可得当 $g(z) \geq 0.5$ 时，有 $z \geq 0$，即 $\theta^TX \geq 0$。

同线性回归模型的不同点在于： $\begin{align*}z \to +\infty, e^{-\infty} \to 0 \Rightarrow g(z)=1 \newline z \to -\infty, e^{\infty}\to \infty \Rightarrow g(z)=0 \end{align*}$

直观一点来个例子，${h_\theta}\left( x \right)=g\left( {\theta_0}+{\theta_1}{x_1}+{\theta_{2}}{x_{2}}\right)$ 是下图模型的假设函数。

![](image/20180111_000814.png)

根据上面的讨论，要进行分类，那么只要 $ {\theta_0}+{\theta_1}{x_1}+{\theta_{2}}{x_{2}}\geq0$ 时，就预测 $y = 1$，即预测为正向类。

如果取 $\theta = \begin{bmatrix} -3\\1\\1\end{bmatrix}$，则有 $z = -3+{x_1}+{x_2}$，当 $z \geq 0$ 即 ${x_1}+{x_2} \geq 3$ 时，易绘制图中的品红色直线即**决策边界**，为正向类（以红叉标注的数据）给出 $y=1$ 的分类预测结果。



上面讨论了逻辑回归模型中线性拟合的例子，下面则是一个多项式拟合的例子，和线性回归中讨论的其实没有多大区别。

为了拟合下图数据，建模多项式假设函数：

${h_\theta}\left( x \right)=g\left( {\theta_0}+{\theta_1}{x_1}+{\theta_{2}}{x_{2}}+{\theta_{3}}x_{1}^{2}+{\theta_{4}}x_{2}^{2} \right)$

这里取 $\theta = \begin{bmatrix} -1\\0\\0\\1\\1\end{bmatrix}$，决策边界对应了一个在原点处的单位圆，如此便可给出分类结果了，如图中品红色曲线：



![](image/20180111_000653.png)

当然，通过一些更为复杂的多项式，还能拟合那些图像显得非常怪异的数据，使得决策边界像碗状、爱心状等等。



简单来说，决策边界就是**分类的分界线**，分类现在实际就由 $z$ (中的 $\theta$)决定啦。

## 6.4 代价函数(Cost Function)



## 6.5 Simplified Cost Function and Gradient Descent

## 6.6 Advanced Optimization

## 6.7 Multiclass Classification_ One-vs-all

# 7 Regularization
## 7.1 The Problem of Overfitting

## 7.2 Cost Function

## 7.3 Regularized Linear Regression

## 7.4 Regularized Logistic Regression