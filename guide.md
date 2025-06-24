# 算法课期末大作业

## 一、实验简介

[cite\_start]本实验涉及如何实现一个图的库,包括图的存储、读写、图结构挖掘算法的实现以及图的可视化。我们希望在实现对应功能时能使用便捷的接口化形式,例如你使用python实现该作业,我们希望你的示例程序如下列格式: [cite: 1]

```python
#以下为一个示例,具体函数名以及结构可与下方不同

#读入文件
g = Graph("输入文件”)
g.save("输出路径")

#实现图结构挖掘算法
g.k_cores("输出结果1")
g.ds("输出结果2")
# 以及其他你实现的算法

#可视化
g.show() #展示图
g.show_coreness() # 展示coreness结构
#..其他你实现的可视化样例
```

## 二、实验内容

### 1\. 图的读写(20分)

图格式如下:

```
n m #表示点数和边数
u v #表示一条u到v的连边
```

  * [cite\_start]学生需要实现一个图的存储结构,支持图的创建、节点和边的添加与删除。 [cite: 1]
      * [cite\_start]C++图的实现可以使用SNAP库 [https://snap.stanford.edu/index.html,也可以自己实现](https://snap.stanford.edu/index.html,也可以自己实现)。 [cite: 1]
          * [cite\_start]此外也可参考部分论文中对图结构的设置: [cite: 1]
            1.  [cite\_start][https://github.com/LijunChang/Cohesive-subgraph-book](https://www.google.com/search?q=https://github.com/LijunChang/Cohesive-subgraph-book) [cite: 1]
            2.  [cite\_start][https://github.com/Xiejiadong/Quantifying-Node-Importance-over-Network-Structural-Stability](https://www.google.com/search?q=https://github.com/Xiejiadong/Quantifying-Node-Importance-over-Network-Structural-Stability). [cite: 1]
      * [cite\_start]Python可以考虑使用NetworkX, igraph, Snap库等,也可以自己实现。 [cite: 2]
        **注:**
          * [cite\_start]自己实现图结构容易拓展到不同算法的相应数据结构,但需花费额外时间。 [cite: 2]
          * [cite\_start]使用库会增加上手难度,大家可以酌情选择,无论是否自己实现均不会影响最终成绩。 [cite: 2]
  * [cite\_start]实现图的读取功能,能够从文件(如:.txt等格式)中读取图的节点和边信息。 [cite: 2]
      * [cite\_start]为了方便处理,我们可以将所有的图都处理为无向简单图,因此你需要做下列步骤: [cite: 2]
          * [cite\_start]图中可能含有重边和自环,你需要对应去除。 [cite: 2]
          * [cite\_start]图中的无向边(u,v)可能表示为(u,v)和(v, u)的有向边形式,你应当对应处理。 [cite:2]
          * [cite\_start]图中顶点可能并不是1-n的全映射,即输入 n=5 时顶点的实际序号可能为(0,1,4,6,9),针对该情况,你需要合理处理,保证输入和输出的顶点能够对应。 [cite: 3]
  * [cite\_start]实现图的基础指标计算。 [cite: 3]
      * [cite\_start]如图的密度,图的平均度等。 [cite: 3]
  * [cite\_start]实现图的写入功能,能够将图的信息输出到文件中。 [cite: 3]

### 2\. 图结构挖掘算法实现(60分)

  * 实现图的结构挖掘算法
    1.  [cite\_start]**k-core分解(10分):** [cite: 3]
          * [cite\_start]需要对读入的图能够计算每个顶点的coreness值,并可以用下面的格式存放到对应输出文件中: [cite: 3]
            ```
            #output.txt
            XXX.XXS # 运行时间
            1 coreness[1]
            2 coreness[2]
            ...
            ```
          * [cite\_start]计算过程可以参考YOJ1127社交网络,也可以查看参考文献[1]。 [cite: 3]
    2.  [cite\_start]**最密子图(15分):** [cite: 3]
          * [cite\_start]**精确算法:** 对输入的图求出最密密度,且输出最密子图对应的子图,如有多个最密子图,可以输出任意一个,输出形式可如下: [cite: 3]
            ```
            #output.txt
            XXX.XXS # 运行时间
            density #最密子图对应的密度
            1 2 3 4 5 6 #最密子图对应的子图
            ```
            [cite\_start]可参考 YOJ 1128 管理公司题解。 [cite: 3]
          * [cite\_start]**近似算法:** 用一个2-近似算法求出2-近似密度子图,所求子图的密度要大于等于最密子图的一半（$\\rho(S)\\ge\\frac{\\rho(S^{\*})}{2}$）。输出形式可如下: [cite: 3]
            ```
            #output.txt
            XXX.XXS # 运行时间
            density #2-近似密度子图对应的密度
            1 2 3 4 5 6 #2-近似密度子图对应的子图
            ```
            [cite\_start]实现方法可参考参考文献[2]中4.2节。 [cite: 3]
    3.  [cite\_start]**k-clique分解(15分):** [cite: 3]
          * [cite\_start]输入对应的k,使用BK算法求k-clique,并将所有极大团输出。输出形式可如下,其中一行对应一个极大团: [cite: 3]
            ```
            #output.txt
            XXX.XXS # 运行时间
            1 2 3 4 5 6 #极大团1
            7 8 9 10 #极大团2
            ```
          * [cite\_start]可参考 YOJ 1125朋友和YOJ 1126 最大团的相应题解。 [cite: 3]
    4.  [cite\_start]**实现下列任一即可获得20分** [cite: 4]
        [cite\_start]以下算法论文中大多有参考代码,可对应参考具体实现。 [cite: 4]
          * **LDS(局部密集子图)**
              * [cite\_start]使用[3]中基于网络流和使用[4]中基于凸优化的算法均可,我们目标是求top-k LDS,其中k是人工输入的参数,输出格式可如下,其中一行为一个LDS。 [cite: 4]
              * [cite\_start]可参考 [https://github.com/chenhao-ma/LDScvx](https://github.com/chenhao-ma/LDScvx) 中实现。 [cite: 4]
                ```
                #output.txt
                XXX.XXS # 运行时间
                density 1 2 3 4 # top-1 LDS的密度和对应顶点
                density 7 8 9 10 # top-2 LDS的密度和对应顶点
                ```
          * **LhCDS(局部h团最稠密子图)**
              * [cite\_start]这是我们在2025年SIGMOD的最新工作[7],检测局部不重叠、接近派系最密集的子图对于社交网络中的社区搜索至关重要。我们的工作引入了一种高效且精确的算法,通过识别 top-k 非重叠、局部h团最稠密子图(LhCDS)来应对这一挑战。实验中可以固定h=3即可。 [cite: 4]
              * [cite\_start]对应复现 [https://github.com/Elssky/IPPV](https://github.com/Elssky/IPPV) 即可。 [cite: 4]
                ```
                #output.txt
                XXX.XXS # 运行时间
                density 1 2 3 4 # top-1 LhCDS的密度和对应顶点
                density 7 8 9 10 # top-2 LhCDS的密度和对应顶点
                ```
          * **k-clique最密子图**
              * [cite\_start]参考[4]中具体实现(论文中任一实现均可),需要人工指定k,输出格式可如下: [cite: 4]
              * [cite\_start]可参考 [https://github.com/btsun/kclistpp](https://github.com/btsun/kclistpp) 中实现。 [cite: 4]
                ```
                #output.txt
                XXX.XXS # 运行时间
                density 1 2 3 4#最密的k-clique子图密度,以及其对应的顶点
                ```
          * **k-core动态维护:**
              * [cite\_start]在实现k-core的基础上,支持对图进行边的插入和边的删除,并能快速计算在边插入和删除后顶点k-core的变化。 [cite: 4]
              * [cite\_start]可参考文献[5]和[6],实现任一即可。 [cite: 4]
              * [cite\_start]每次输入一条边(u,v),同时输入其为插入边或删除边,插入和删除后的输出结果和k-core相同。 [cite: 4]
                ```
                #output.txt
                XXX.XXS # 运行时间
                1 coreness[1]
                2 coreness[2]
                ...
                ```
          * [cite\_start]**k-vcc分解:** [cite: 4]
              * [cite\_start]**k-vcc定义:** 1) 移除图G中任意小于等于k-1个顶点,G都不会断开连接 2) 该图是极大的,即不存在一个图G'是图G的超集,并且G'也满足性质1)。 [cite: 5]
              * [cite\_start]可参考论文: [cite: 5]
                  * [cite\_start]Enumerating k-Vertex Connected Components in Large (ICDE 2019) (精确算法)、这篇论文提出了自上而下基于图划分的k-vcc查找方式(难度★★★)。 [cite: 5]
                  * [cite\_start]Towards k-vertex connected component discovery from large networks (WWW)(近似算法)、这篇论文提出了自下而上基于种子扩展的k-vcc查找方式 (难度★★★)。 [cite: 5]
              * [cite\_start]以上两种方式选择其一即可。 [cite: 5]
              * [cite\_start]输入对应的k,使用最大流算法求k-vcc,并将所有k-vcc输出。输出形式如下,其中一行对应一个k-vcc: [cite: 5]
                ```
                #output.txt
                XXX.XXS # 运行时间
                1 2 3 4 5 6 # k-vcc1
                7 8 9 10 # k-vcc2
                ```

### 3\. 图可视化 (20分)

#### 3.1 总体要求

  * [cite\_start]实现一个简单的图可视化功能,能够将图以图形的方式展示在屏幕上。 [cite: 5]
  * [cite\_start]可视化应支持节点和边的样式设置,如颜色、大小、标签等。 [cite: 5]
  * [cite\_start]可视化可以考虑实现良好的交互性,如缩放、平移、布局调整等。 [cite: 5]
  * [cite\_start]对各个图挖掘算法的结果也可以进行相应的可视化展示。 [cite: 5]

#### 3.2 推荐图可视化库

  * [cite\_start]**C++可选(难度:★★★)** [cite: 5]

      * [cite\_start]**Graphviz:** 官网链接 [https://graphviz.org/](https://graphviz.org/); [cite: 5]
      * [cite\_start]Graphviz (Graph Visualization Software)是一个由AT\&T实验室启动的开源工具包,专门用于绘制图形的布局和渲染。它广泛应用于自动图形布局处理,支持多种类型的图形表示,包括有向图和无向图。 [cite: 6]
      * [cite\_start]官方教程: [https://www.graphviz.org/pdf/libguide.pdf](https://www.graphviz.org/pdf/libguide.pdf) [cite: 6]
      * [cite\_start]Library Usage: [https://graphviz.org/docs/library/](https://graphviz.org/docs/library/) [cite: 6]
      * [cite\_start]官方库: [https://gitlab.com/graphviz/graphviz](https://gitlab.com/graphviz/graphviz) [cite: 6]
      * [cite\_start]参考博客: [https://blog.csdn.net/root\_clive/article/details/122395524](https://blog.csdn.net/root_clive/article/details/122395524) [cite: 6]
      * [cite\_start]一些例子: [cite: 6]
          * [cite\_start][demo.c](https://www.graphviz.org/dot.demo/demo.c) [cite: 6]
          * [cite\_start][dot.c](https://www.graphviz.org/dot.demo/dot.c) [cite: 6]
          * [cite\_start][example.c](https://www.graphviz.org/dot.demo/example.c) [cite: 6]
          * [cite\_start][simple.c](https://www.graphviz.org/dot.demo/simple.c) [cite: 6]
          * [cite\_start][Makefile](https://www.graphviz.org/dot.demo/Makefile) [cite: 6]
      * [cite\_start][https://graphviz.org/Gallery/twopi/twopi2.html](https://graphviz.org/Gallery/twopi/twopi2.html) [cite: 6]
      * **效果展示:**
        [cite\_start]*(Image from source [cite: 7])*

  * [cite\_start]**Python可选(难度:★★)** [cite: 7]

      * [cite\_start]**networkx:** 官方教程: [https://www.osgeo.cn/networkx/tutorial.html\#drawing-graphs](https://www.google.com/search?q=https://www.osgeo.cn/networkx/tutorial.html%23drawing-graphs) [cite: 7]
      * 参考博客:
          * [cite\_start][https://zhuanlan.zhihu.com/p/381645334](https://zhuanlan.zhihu.com/p/381645334) [cite: 7]
          * [cite\_start][https://zhuanlan.zhihu.com/p/36700425](https://zhuanlan.zhihu.com/p/36700425) [cite: 7]
          * [cite\_start][https://www.cnblogs.com/luohenyueji/p/16991239.html](https://www.cnblogs.com/luohenyueji/p/16991239.html) [cite: 7]
      * **效果展示:**
        [cite\_start]*(Image from source [cite: 8])*

## 三、实验要求

  * [cite\_start]使用C++或Python语言进行编程。 [cite: 8]
  * [cite\_start]代码应具有良好的可读性,注释清晰。 [cite: 8]
  * [cite\_start]需要提交完整的代码和实验报告,报告中应包含实验的设计思路、算法实现、测试结果和分析等内容。 [cite: 8]
  * [cite\_start]鼓励进行创新,可以尝试实现额外的图算法或优化现有算法。 [cite: 8]

## 四、提交材料

  * [cite\_start]完整的源代码文件。 [cite: 8]
  * [cite\_start]实验报告,包括实验目的、实验内容、实验步骤、实验结果及分析等。 [cite: 8]
      * [cite\_start]可以贴部分代码并解释代码逻辑。 [cite: 8]
  * [cite\_start]运行结果文件: [cite: 8]
      * [cite\_start]其中包含你所实现的图挖掘算法在下方三个数据集上的运行结果。 [cite: 8]

## 五、提供材料

  * [cite\_start]**数据集:** 社交网络,基于位置的网络等。 [cite: 8]
    [cite\_start]以下三个数据集均为未处理的版本,你需要参考第一部分中的描述对数据集加以处理。 [cite: 8]
      * [cite\_start]Gowalla.txt [cite: 8]
      * [cite\_start]Amazon.txt [cite: 8]
      * [cite\_start]CondMat.txt [cite: 8]

| dataset | nodes | edges | Average Clustering Coefficient | Diameter (Longest shortest path) |
| :--- | :--- | :--- | :--- | :--- |
| Gowalla | 196591 | 950327 | 0.2367 | 14 |
| Amazon | 334863 | 925872 | 0.3967 | 44 |
| CondMat | 23133 | 93497 | 0.6334 | 14 |
[cite\_start]*Table data from source [cite: 9]*

**注:**

1.  [cite\_start]由于不同同学的电脑配置可能不同,下方的数据集如果无法全部在本地运行,只运行部分即可; [cite: 10]
2.  [cite\_start]如果全部数据集均无法运行(超时或内存过大),可自行构造小数据集,对应给出数据集的信息描述即可,如上方所示。 [cite: 10]

## 参考文献

[cite\_start][1] Batagelj V, Zaversnik M. An o (m) algorithm for cores decomposition of networks[J]. arXiv preprint cs/0310049, 2003. [cite: 10, 11]
[cite\_start][2] Subgraph C. Cohesive Subgraph Computation over Large Sparse Graphs[J]. [cite: 11]
[3] Qin L, Li RH, Chang L, et al. Locally densest subgraph discovery[C]//Proceedings of the 21th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. [cite\_start]2015: 965-974. [cite: 12]
[4] Sun B, Danisch M, Chan TH H, et al. Kclist++: A simple algorithm for finding k-clique densest subgraphs in large graphs[J]. [cite\_start]Proceedings of the VLDB Endowment (PVLDB), 2020. [cite: 13, 14, 15]
[5] Zhang Y, Yu J X, Zhang Y, et al. A fast order-based approach for core maintenance[C]//2017 IEEE 33rd International Conference on Data Engineering (ICDE). [cite\_start]IEEE, 2017: 337-348. [cite: 16]
[6] LI RH, Yu J X, Mao R. Efficient core maintenance in large dynamic graphs[J]. [cite\_start]IEEE transactions on knowledge and data engineering, 2013, 26(10): 2453-2465. [cite: 17, 18]
[7] Xu X, Liu H, Lv X, et al. An Efficient and Exact Algorithm for Locally h-Clique Densest Subgraph Discovery[J]. [cite\_start]Proceedings of the ACM on Management of Data, 2024, 2(6): 1-26. [cite: 19, 20]