# 2 单变量线性回归（Linear Regression with One Variable）

## 2.1 模型表示（Model Representation）

1. 房价预测训练集
| Size in $feet^2$ ($x$) | Price (\$) in 1000's($y$) |
| ---------------------- | ------------------------- |
| 2104                   | 460                       |
| 1416                   | 232                       |
| 1534                   | 315                       |
| 852                    | 178                       |
| ...                    | ...                       |

房价预测训练集中，同时给出了输入 $x$ 和输出结果 $y$，即给出了人为标注的**”正确结果“**，且预测的量是连续的，属于监督学习中的回归问题。

2. **问题解决模型**

![](/images/20180105_212048.png)

其中 $h$ 代表结果函数，也称为**假设（hypothesis）** 。假设函数根据输入（房屋的面积），给出预测结果输出（房屋的价格），即是一个 $X\to Y$ 的映射。

$h_\theta(x)=\theta_0+\theta_1x$，为解决房价问题的一种可行表达式。

> $x$: 特征/输入变量。

上式中，$\theta$ 为参数，$\theta$ 的变化才决定了输出结果，不同以往，这里的 $x$ 被我们**视作已知**（不论是数据集还是预测时的输入），所以怎样解得 $\theta$ 以更好地拟合数据，成了求解该问题的最终问题。

单变量，即只有一个特征（如例子中房屋的面积这个特征）。

## 2.2 代价函数（Cost Function）

> 李航《统计学习方法》一书中，损失函数与代价函数两者为**同一概念**，未作细分区别，全书没有和《深度学习》一书一样混用，而是统一使用**损失函数**来指代这类类似概念。
>
> 吴恩达（Andrew Ng）老师在其公开课中对两者做了细分。**如果要听他的课做作业，不细分这两个概念是会被打小手扣分的**！这也可能是因为老师发现了业内混用的乱象，想要治一治吧。
>
> **损失函数**（Loss/Error Function）: 计算**单个**样本的误差。[link](https://www.coursera.org/learn/neural-networks-deep-learning/lecture/yWaRd/logistic-regression-cost-function)
>
> **代价函数**（Cost Function）: 计算整个训练集**所有损失函数之和的平均值**
>
> 
>
> 综合考虑，本笔记对两者概念进行细分，若有所谬误，欢迎指正。
>
> [机器学习中的目标函数、损失函数、代价函数有什么区别？- 知乎](https://www.zhihu.com/question/52398145/answer/298003145)



我们的目的在于求解预测结果 $h$ 最接近于实际结果 $y$ 时 $\theta$ 的取值，则问题可表达为**求解 $\sum\limits_{i=0}^{m}(h_\theta(x^{(i)})-y^{(i)})$ 的最小值**。

> $m$: 训练集中的样本总数
>
> $y$: 目标变量/输出变量
>
> $\left(x, y\right)$: 训练集中的实例
>
> $\left(x^{\left(i\right)},y^{\left(i\right)}\right)$: 训练集中的第 $i$ 个样本实例

![](/images/20180105_224648.png)

上图展示了当 $\theta$ 取不同值时，$h_\theta\left(x\right)$ 对数据集的拟合情况，蓝色虚线部分代表**建模误差**（预测结果与实际结果之间的误差），我们的目标就是最小化所有误差之和。

为了求解最小值，引入代价函数（Cost Function）概念，用于度量建模误差。考虑到要计算最小值，应用二次函数对求和式建模，即应用统计学中的平方损失函数（最小二乘法）：

$$J(\theta_0,\theta_1)=\dfrac{1}{2m}\displaystyle\sum_{i=1}^m\left(\hat{y}_{i}-y_{i} \right)^2=\dfrac{1}{2m}\displaystyle\sum_{i=1}^m\left(h_\theta(x_{i})-y_{i}\right)^2$$ 

> $\hat{y}$: $y$ 的预测值
>
> 系数 $\frac{1}{2}$ 存在与否都不会影响结果，这里是为了在应用梯度下降时便于求解，平方的导数会抵消掉 $\frac{1}{2}$ 。

讨论到这里，我们的问题就转化成了**求解 $J\left( \theta_0, \theta_1  \right)$ 的最小值**。

## 2.3 代价函数 - 直观理解1（Cost Function - Intuition I）

根据上节视频，列出如下定义：

- 假设函数（Hypothesis）: $h_\theta(x)=\theta_0+\theta_1x$
- 参数（Parameters）: $\theta_0, \theta_1$
- 代价函数（Cost Function）: $J\left( \theta_0, \theta_1  \right)=\frac{1}{2m}\sum\limits_{i=1}^{m}{{{\left( {{h}_{\theta }}\left( {{x}^{(i)}} \right)-{{y}^{(i)}} \right)}^{2}}}$
- 目标（Goal）: $\underset{\theta_0, \theta_1}{\text{minimize}} J \left(\theta_0, \theta_1 \right)$

为了直观理解代价函数到底是在做什么，先假设 $\theta_0 = 0$，并假设训练集有三个数据，分别为$\left(1, 1\right), \left(2, 2\right), \left(3, 3\right)$，这样在平面坐标系中绘制出 $h_\theta\left(x\right)$ ，并分析 $J\left(\theta_0, \theta_1\right)$ 的变化。

![](/images/20180106_085915.png)

右图 $J\left(\theta_0, \theta_1\right)$ 随着 $\theta_1$ 的变化而变化，可见**当 $\theta_1 = 1$ 时，$J\left(\theta_0, \theta_1 \right) = 0$，取得最小值，**对应于左图青色直线，即函数 $h$ 拟合程度最好的情况。

## 2.4 代价函数 - 直观理解2（Cost Function - Intuition II）

> 注：该部分由于涉及到了多变量成像，可能较难理解，要求只需要理解上节内容即可，该节如果不能较好理解可跳过。

给定数据集：

![](/images/20180106_091307.png)

参数在 $\theta_0$ 不恒为 $0$ 时代价函数 $J\left(\theta\right)$ 关于 $\theta_0, \theta_1$ 的3-D图像，图像中的高度为代价函数的值。

![](/images/20180106_090904.png)

由于3-D图形不便于标注，所以将3-D图形转换为**轮廓图（contour plot）**，下面用轮廓图（下图中的右图）来作直观理解，其中相同颜色的一个圈代表着同一高度（同一 $J\left(\theta\right)$ 值）。

$\theta_0 = 360, \theta_1 =0$ 时：

![](/images/0f38a99c8ceb8aa5b90a5f12136fdf43.png)

大概在 $\theta_0 = 0.12, \theta_1 =250$ 时：

![](/images/20180106_092119.png)

上图中最中心的点（红点），近乎为图像中的最低点，也即代价函数的最小值，此时对应 $h_\theta\left(x\right)$ 对数据的拟合情况如左图所示，嗯，一看就拟合的很不错，预测应该比较精准啦。

## 2.5 梯度下降（Gradient Descent）

在特征量很大的情况下，即便是借用计算机来生成图像，人工的方法也很难读出 $J\left(\theta\right)$ 的最小值，并且大多数情况无法进行可视化，故引入**梯度下降（Gradient Descent）方法，让计算机自动找出最小化代价函数时对应的 $\theta$ 值。**

梯度下降背后的思想是：开始时，我们随机选择一个参数组合$\left( {\theta_{0}},{\theta_{1}},......,{\theta_{n}} \right)$即起始点，计算代价函数，然后寻找下一个能使得代价函数下降最多的参数组合。不断迭代，直到找到一个**局部最小值（local minimum）**，由于下降的情况只考虑当前参数组合周围的情况，所以无法确定当前的局部最小值是否就是**全局最小值（global minimum）**，不同的初始参数组合，可能会产生不同的局部最小值。

下图根据不同的起始点，产生了两个不同的局部最小值。

![](/images/db48c81304317847870d486ba5bb2015.jpg)

视频中举了下山的例子，即我们在山顶上的某个位置，为了下山，就不断地看一下周围**下一步往哪走**下山比较快，然后就**迈出那一步**，一直重复，直到我们到达山下的某一处**陆地**。

梯度下降公式：

$$
\begin{align*}
& \text{Repeat until convergence:} \; \lbrace \\
&{{\theta }_{j}}:={{\theta }_{j}}-\alpha \frac{\partial }{\partial {{\theta }_{j}}}J\left( {\theta_{0}},{\theta_{1}}  \right) \\
\rbrace
\end{align*}
$$


> ${\theta }_{j}$: 第 $j$ 个特征参数
>
> ”:=“: 赋值操作符
>
> $\alpha$: 学习速率（learning rate）, $\alpha > 0$
> 
> $\frac{\partial }{\partial {{\theta }_{j}}}J\left( \theta_0, \theta_1  \right)$: $J\left( \theta_0, \theta_1 \right)$ 的偏导

公式中，学习速率决定了参数值变化的速率即”**走多少距离**“，而偏导这部分决定了下降的方向即”**下一步往哪里**“走（当然实际上的走多少距离是由偏导值给出的，学习速率起到调整后决定的作用），收敛处的局部最小值又叫做极小值，即”**陆地**“。

![](/images/20180106_101659.png)

注意，在计算时要**批量更新 $\theta$ 值**，即如上图中的左图所示，否则结果上会有所出入，原因不做细究。

## 2.6 梯度下降直观理解（Gradient Descent Intuition）

该节探讨 $\theta_1$ 的梯度下降更新过程，即 $\theta_1 := \theta_1 - \alpha\frac{d}{d\theta_1}J\left(\theta_1\right)$，此处为了数学定义上的精确性，用的是 $\frac{d}{d\theta_1}J\left(\theta_1\right)$，如果不熟悉微积分学，就把它视作之前的 $\frac{\partial}{\partial\theta}$ 即可。

![](/images/20180106_184926.png)

把红点定为初始点，切于初始点的红色直线的斜率，表示了函数 $J\left(\theta\right)$ 在初始点处有**正斜率**，也就是说它有**正导数**，则根据梯度下降公式 ，${{\theta }_{j}}:={{\theta }_{j}}-\alpha \frac{\partial }{\partial {{\theta }_{j}}}J\left( \theta_0, \theta_1  \right)$ 右边的结果是一个正值，即 $\theta_1$ 会**向左边移动**。这样不断重复，直到收敛（达到局部最小值，即斜率为0）。

初始 $\theta$ 值（初始点）是任意的，若初始点恰好就在极小值点处，梯度下降算法将什么也不做（$\theta_1 := \theta_1 - \alpha*0$）。

> 不熟悉斜率的话，就当斜率的值等于图中三角形的高度除以水平长度好啦，精确地求斜率的方法是求导。



对于学习速率 $\alpha$，需要选取一个合适的值才能使得梯度下降算法运行良好。

- 学习速率过小图示：

  ![](/images/20180106_190944.png)

  收敛的太慢，需要更多次的迭代。

- 学习速率过大图示：

  ![](/images/20180106_191023.png)

  可能越过最低点，甚至导致无法收敛。

**学习速率只需选定即可**，不需要在运行梯度下降算法的时候进行动态改变，随着斜率越来越接近于0，代价函数的变化幅度会越来越小，直到收敛到局部极小值。

如图，品红色点为初始点，代价函数随着迭代的进行，变化的幅度越来越小。

![](/images/20180106_191956.png)

**最后，梯度下降不止可以用于线性回归中的代价函数，还通用于最小化其他的代价函数。**

## 2.7 线性回归中的梯度下降（Gradient Descent For Linear Regression）

线性回归模型

- $h_\theta(x)=\theta_0+\theta_1x$
- $J\left( \theta_0, \theta_1  \right)=\frac{1}{2m}\sum\limits_{i=1}^{m}{{{\left( {{h}_{\theta }}\left( {{x}^{(i)}} \right)-{{y}^{(i)}} \right)}^{2}}}$

梯度下降算法
$$
\begin{align*}
  & \text{Repeat until convergence:} \; \lbrace \\
  &{{\theta }_{j}}:={{\theta }_{j}}-\alpha \frac{\partial }{\partial {{\theta }_{j}}}J\left( {\theta_{0}},{\theta_{1}}  \right) \\
  \rbrace
  \end{align*}
$$



直接将线性回归模型公式代入梯度下降公式可得出公式

![](/images/20180106_203726.png)

当 $j = 0, j = 1$ 时，**线性回归中代价函数求导的推导过程（看不懂请查阅导数基本公式）：**
$$
\begin{align*}
\frac{\partial}{\partial\theta_j} J(\theta_1, \theta_2)&=\frac{\partial}{\partial\theta_j} \left(\frac{1}{2m}\sum\limits_{i=1}^{m}{{\left( {{h}_{\theta }}\left( {{x}^{(i)}} \right)-{{y}^{(i)}} \right)}^{2}} \right)\\
&=\left(\frac{1}{2m}*2\sum\limits_{i=1}^{m}{{\left( {{h}_{\theta }}\left( {{x}^{(i)}} \right)-{{y}^{(i)}} \right)}} \right)*\frac{\partial}{\partial\theta_j}{{\left( {{h}_{\theta }}\left( {{x}^{(i)}} \right)-{{y}^{(i)}} \right)}}\\
&=\left(\frac{1}{m}\sum\limits_{i=1}^{m}{{\left( {{h}_{\theta }}\left( {{x}^{(i)}} \right)-{{y}^{(i)}} \right)}} \right)*\frac{\partial}{\partial\theta_j}{{\left(\theta_0{x_0^{(i)}} + \theta_1{x_1^{(i)}}-{{y}^{(i)}} \right)}}
\end{align*}
$$


所以当 $j = 0$ 时：

$$
\frac{\partial}{\partial\theta_0} J(\theta)=\frac{1}{m}\sum\limits_{i=1}^{m}{{\left( {{h}_{\theta }}\left( {{x}^{(i)}} \right)-{{y}^{(i)}} \right)}} *x_0^{(i)}
$$


所以当 $j = 1$ 时：

$$
\frac{\partial}{\partial\theta_1} J(\theta)=\frac{1}{m}\sum\limits_{i=1}^{m}{{\left( {{h}_{\theta }}\left( {{x}^{(i)}} \right)-{{y}^{(i)}} \right)}} *x_1^{(i)}
$$
上文中所提到的梯度下降，都为批量梯度下降（Batch Gradient Descent），即每次计算都使用**所有**的数据集 $\left(\sum\limits_{i=1}^{m}\right)$ 更新。

由于线性回归函数呈现**碗状**，且**只有一个**全局的最优值，所以函数**一定总会**收敛到全局最小值（学习速率不可过大）。同时，函数 $J$ 被称为**凸二次函数**，而线性回归函数求解最小值问题属于**凸函数优化问题**。

![](/images/24e9420f16fdd758ccb7097788f879e7.png)

另外，使用循环求解，代码较为冗余，后面会讲到如何使用**向量化（Vectorization）**来简化代码并优化计算，使梯度下降运行的更快更好。
