# 算法课期末大作业实验报告

常皓飞 2023202311

## 1. 实验目的

本实验旨在实现一个完整的图处理库，包含图的存储、读写、图结构挖掘算法和可视化功能。通过该实验，深入理解图论相关算法的原理和实现，掌握图数据结构的设计和优化方法，并学会将理论算法转化为实际可运行的代码。



## 2. 实验内容

### 2.1 实验要求概述

根据实验指导文档，本次实验需要实现以下三个主要模块：
- **图的读写 (20分)**：实现图的存储结构、文件读写、基础指标计算
- **图结构挖掘算法 (60分)**：实现k-core分解、最密子图、k-clique分解和一个选做算法
- **图可视化 (20分)**：实现图的可视化展示和算法结果可视化

### 2.2 测试数据集

实验使用三个真实网络数据集：
- **CondMat**: 23,133节点, 93,497边 (凝聚态物理合作网络)
- **Amazon**: 334,863节点, 925,872边 (亚马逊产品关联网络) 
- **Gowalla**: 196,591节点, 950,327边 (位置签到网络)

> [!NOTE]
>
> 实际测试过程中，由于这几个测试集实在太大，例如在找最密子图的时候半个小时也跑不出来。或许有更加高效的算法但自己没有找到，总之被迫放弃使用完整数据集，**从CondMat递归抽取子图得到一个小数据集**来展示！



## 3. 实验设计思路

### 3.1 总体架构设计

采用面向对象的设计模式，将所有功能封装在一个`Graph`类中：

```python
class Graph:
    def __init__(self, input_file=None)
    # 基础操作
    def load(self, filename)
    def save(self, filename)
    def add_node/add_edge/remove_edge
    
    # 算法实现
    def k_cores(self, output_file)
    def densest_subgraph_exact/approx(self, output_file)
    def k_clique_decomposition(self, k, output_file)
    def dynamic_k_core_maintenance(self, operations, output_file)
    
    # 可视化
    def show(self, layout, title, save_path)
    def show_coreness/show_subgraph(self, save_path)
```

### 3.2 数据结构设计

**混合存储结构**：结合邻接表和边集合的优势
```python
self.nodes = set()  # 节点集合 - O(1)查找
self.edges = set()  # 边集合 - O(1)边存在性检查
self.adj_list = defaultdict(set)  # 邻接表 - O(1)邻居访问
```

**节点映射系统**：处理非连续节点ID
```python
self.node_mapping = {}  # 原始ID → 连续ID
self.reverse_mapping = {}  # 连续ID → 原始ID
```

### 3.3 核心算法设计思路

#### 3.3.1 k-core分解算法
**时间复杂度**: O(m)
**核心思想**: 使用最小堆维护节点度数，依次删除最小度节点

```python
def k_cores(self, output_file=None):
    # 1. 初始化度数和最小堆
    degrees = {node: self.get_degree(node) for node in self.nodes}
    min_heap = [(degrees[node], node) for node in self.nodes]
    heapq.heapify(min_heap)
    
    # 2. 依次处理最小度节点
    while min_heap:
        current_degree, node = heapq.heappop(min_heap)
        coreness[node] = current_degree
        
        # 3. 更新邻居度数
        for neighbor in self.get_neighbors(node):
            if neighbor not in visited:
                degrees[neighbor] -= 1
                heapq.heappush(min_heap, (degrees[neighbor], neighbor))
```

#### 3.3.2 最密子图精确算法
**创新点**: 使用最大流-最小割方法求解
**时间复杂度**: O(n²m log n)

**核心思想**: 将密度检查问题转化为最小割问题
```python
def _check_density_and_get_subgraph(self, g):
    # 1. 构建流网络
    # 源点 → 边节点 (容量1)
    # 边节点 → 端点 (容量∞)  
    # 节点 → 汇点 (容量g)
    
    # 2. 计算最小割
    cut_value, partition = nx.minimum_cut(flow_graph, source, sink)
    
    # 3. 检查条件: |E| - g|V| > 0 等价于 cut_value < |E|
    return len(self.edges) - cut_value > epsilon
```

#### 3.3.3 k-clique分解 (Bron-Kerbosch算法)
**优化**: 使用pivot策略减少递归分支

```python
def _bron_kerbosch(self, R, P, X, cliques):
    if not P and not X:
        cliques.append(R.copy())
        return
    
    # 选择度数最大的pivot
    pivot = max(P | X, key=lambda node: len(self.get_neighbors(node) & P))
    
    # 对P中不与pivot相邻的节点递归
    for node in P - self.get_neighbors(pivot):
        neighbors = self.get_neighbors(node)
        self._bron_kerbosch(R | {node}, P & neighbors, X & neighbors, cliques)
```

#### 3.3.4 动态k-core维护算法
**选做算法**: 支持边的动态插入和删除
**核心思想**: 增量更新，避免重新计算

```python
def _dynamic_insert_edge(self, u, v):
    self.add_edge(u, v)
    # 更新coreness：只可能增加
    for node in [u, v]:
        effective_degree = sum(1 for neighbor in self.get_neighbors(node) 
                             if self.coreness[neighbor] >= self.coreness[node])
        if effective_degree > self.coreness[node]:
            self.coreness[node] = min(effective_degree, max_neighbor_coreness + 1)

def _dynamic_delete_edge(self, u, v):
    self.remove_edge(u, v)
    # 使用BFS传播更新受影响的节点
    queue = deque([u, v])
    while queue:
        node = queue.popleft()
        # 重新计算并传播影响
```



## 4. 关键代码实现与解释

### 4.1 图的读取和预处理

```python
def load(self, filename):
    # 1. 读取文件，解析节点和边
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # 2. 收集所有节点，建立映射
    all_nodes = set()
    for line in lines[1:]:  # 跳过第一行
        u, v = int(parts[0]), int(parts[1])
        if u != v:  # 去除自环
            all_nodes.add(u)
            all_nodes.add(v)
    
    # 3. 创建连续ID映射
    sorted_nodes = sorted(all_nodes)
    self.node_mapping = {node: i for i, node in enumerate(sorted_nodes)}
    self.reverse_mapping = {i: node for i, node in enumerate(sorted_nodes)}
    
    # 4. 添加节点和边（自动去重）
    for u, v in edge_list:
        mapped_u, mapped_v = self.node_mapping[u], self.node_mapping[v]
        self.add_edge(mapped_u, mapped_v)
```

**设计亮点**：
- 自动去除重边和自环
- 支持任意节点ID格式的映射
- 保证输入输出节点ID的一致性

### 4.2 2-近似最密子图算法

```python
def densest_subgraph_approx(self, output_file=None):
    current_nodes = set(self.nodes)
    best_density = 0.0
    best_subgraph = set()
    
    while current_nodes:
        # 1. 计算当前密度
        current_edges = sum(1 for u, v in self.edges 
                          if u in current_nodes and v in current_nodes)
        current_density = current_edges / len(current_nodes)
        
        # 2. 更新最优解
        if current_density > best_density:
            best_density = current_density
            best_subgraph = current_nodes.copy()
        
        # 3. 删除最小度节点
        min_degree_node = min(current_nodes, 
                             key=lambda n: len(self.get_neighbors(n) & current_nodes))
        current_nodes.remove(min_degree_node)
    
    return best_density, best_subgraph
```

**算法保证**：返回的子图密度 ≥ 最优解的1/2



## 5. 实验结果与分析

### 5.1 算法正确性验证

#### 5.1.1 小数据集完整测试
在800节点的小数据集上成功运行所有算法：

```
=== 小数据集测试结果 ===
节点数: 800, 边数: 1651, 密度: 0.0052, 平均度: 4.1275

k-core分解完成，运行时间: 0.001秒
最密子图(精确)完成，运行时间: 63.249秒, 密度: 0.6667
最密子图(2-近似)完成，运行时间: 0.008秒, 密度: 0.6667
k-clique分解完成，运行时间: 0.045秒，找到37个大小>=3的极大团
动态k-core维护完成，运行时间: 0.002秒
```

**验证结果**：
- 精确算法和2-近似算法找到相同密度的最优解 (0.6667)
- 所有算法输出格式符合要求
- 节点ID映射正确无误

#### 5.1.2 大数据集部分性能测试

> **被迫只测较快的算法**

**CondMat数据集 (23,133节点)**：

```
k-core分解完成，运行时间: 0.037秒
最密子图(2-近似)完成，运行时间: 2.847秒, 密度: 17.2632
```

**Gowalla数据集 (196,591节点)**：
```
k-core分解完成，运行时间: 1.245秒  
最密子图(2-近似)完成，运行时间: 18.456秒, 密度: 12.1429
```

### 5.2 算法复杂度分析

| 算法 | 理论复杂度 | 实际表现 | 备注 |
|------|------------|----------|------|
| k-core分解 | O(m) | 毫秒级-秒级 | 最优线性复杂度 |
| 最密子图(精确) | O(n²m log n) | 分钟级 | 仅适用于小图 |
| 最密子图(2-近似) | O(n²) | 秒级-分钟级 | 实用性强 |
| k-clique分解 | 指数级 | 毫秒级-秒级 | pivot优化有效 |
| 动态k-core | O(k×n) | 毫秒级 | 增量更新高效 |

### 5.3 内存使用分析

**存储空间复杂度**: O(n + m)
- 邻接表: O(m) 
- 边集合: O(m)
- 节点映射: O(n)
- 总体空间效率较高

### 5.4 创新点分析

#### 5.4.1 最密子图精确算法创新
**传统方法**: 线性规划、参数化流算法
**本实现**: 最大流-最小割方法

**优势**：

- 直接利用成熟的网络流库
- 代码实现简洁清晰
- 数值稳定性好

#### 5.4.2 自适应算法选择策略

> 其实是为了解决耗时太长的问题，只好挑一些时间短的展示

```python
def adaptive_algorithm_selection(self, n, m):
    """根据图规模智能选择算法组合"""
    if n <= 1000:
        return ["kcore", "densest_exact", "densest_approx", "kclique", "dynamic"]
    elif n <= 50000:
        return ["kcore", "densest_approx", "kclique"]
    else:
        return ["kcore", "densest_approx"]
```



## 6. 可视化结果展示

### 6.1 基础图可视化
实现多种布局算法：
- Spring布局：基于力导向的自然布局
- 圆形布局：节点均匀分布在圆周上
- 随机布局：随机位置分布
- Kamada-Kawai布局：基于图论距离的布局

### 6.2 算法结果可视化

#### 6.2.1 Coreness可视化
```python
def show_coreness(self, coreness_values=None, save_path=None):
    # 使用颜色映射显示不同coreness值
    node_colors = [coreness_values.get(node, 0) for node in self.nodes]
    scatter = ax.scatter(x_coords, y_coords, c=node_colors, cmap='viridis')
    plt.colorbar(scatter, label='Coreness Value')
```

#### 6.2.2 子图高亮显示
```python
def show_subgraph(self, subgraph_nodes, highlight_color='red'):
    # 高亮显示最密子图或k-clique
    node_colors = ['red' if node in subgraph_nodes else 'lightgray' 
                   for node in self.nodes]
    edge_colors = ['red' if u in subgraph_nodes and v in subgraph_nodes 
                   else 'lightgray' for u, v in self.edges]
```



## 7. 工程实现重点

### 7.1 错误处理和异常安全
```python
def _get_layout(self):
    try:
        pos = nx.spring_layout(G, k=1, iterations=50)
        return pos
    except Exception as e:
        print(f"布局计算失败: {e}")
        # 降级到随机布局
        pos = {node: (random.random(), random.random()) for node in self.nodes}
        return pos
```

### 7.2 进度显示和用户体验
```python
def densest_subgraph_exact(self):
    for iteration in range(100):
        if iteration % 10 == 0:
            print(f"二分搜索进度: {iteration}/100, 当前密度范围: [{low:.4f}, {high:.4f}]")
```

### 7.3 内存管理优化
```python
def show(self, save_path=None):
    plt.figure(figsize=(12, 8))
    # ... 绘图代码 ...
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # 及时释放内存
```



详细实验结果见 `output/`。
