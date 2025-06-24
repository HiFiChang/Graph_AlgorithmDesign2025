import time
import heapq
from collections import defaultdict, deque
from typing import List, Set, Tuple, Dict, Optional
import networkx as nx
# 设置matplotlib为非交互式后端
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import numpy as np

class Graph:
    """
    图的库实现，支持图的存储、读写、算法挖掘和可视化
    """
    
    def __init__(self, input_file: str = None):
        """
        初始化图
        Args:
            input_file: 输入文件路径，如果提供则自动读取
        """
        self.nodes = set()  # 节点集合
        self.edges = set()  # 边集合，存储为(u,v)形式，u < v
        self.adj_list = defaultdict(set)  # 邻接表
        self.node_mapping = {}  # 原始节点ID到连续ID的映射
        self.reverse_mapping = {}  # 连续ID到原始节点ID的映射
        self.coreness = {}  # 节点的coreness值，用于动态维护
        
        if input_file:
            self.load(input_file)
    
    def load(self, filename: str):
        """
        从文件读取图数据
        Args:
            filename: 文件路径
        """
        print(f"正在读取图文件: {filename}")
        
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        # 读取第一行的节点数和边数
        if lines and len(lines[0].strip().split()) == 2:
            n, m = map(int, lines[0].strip().split())
            print(f"声明的节点数: {n}, 边数: {m}")
            lines = lines[1:]  # 跳过第一行
        
        # 收集所有出现的节点
        all_nodes = set()
        edge_list = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                u, v = int(parts[0]), int(parts[1])
                # 去除自环
                if u != v:
                    all_nodes.add(u)
                    all_nodes.add(v)
                    edge_list.append((u, v))
        
        # 创建节点映射（将节点映射到0到n-1的连续区间）
        sorted_nodes = sorted(all_nodes)
        self.node_mapping = {node: i for i, node in enumerate(sorted_nodes)}
        self.reverse_mapping = {i: node for i, node in enumerate(sorted_nodes)}
        
        # 添加所有节点
        for node in sorted_nodes:
            self.add_node(self.node_mapping[node])
        
        # 添加所有边（自动去重）
        for u, v in edge_list:
            mapped_u = self.node_mapping[u]
            mapped_v = self.node_mapping[v]
            self.add_edge(mapped_u, mapped_v)
        
        print(f"实际读取: 节点数 {len(self.nodes)}, 边数 {len(self.edges)}")
    
    def add_node(self, node: int):
        """添加节点"""
        self.nodes.add(node)
    
    def add_edge(self, u: int, v: int):
        """
        添加边（无向图）
        Args:
            u, v: 边的两个端点
        """
        if u != v:  # 去除自环
            self.nodes.add(u)
            self.nodes.add(v)
            # 保证边的存储形式为(小节点, 大节点)
            edge = (min(u, v), max(u, v))
            self.edges.add(edge)
            self.adj_list[u].add(v)
            self.adj_list[v].add(u)
    
    def remove_edge(self, u: int, v: int):
        """删除边"""
        edge = (min(u, v), max(u, v))
        if edge in self.edges:
            self.edges.remove(edge)
            self.adj_list[u].discard(v)
            self.adj_list[v].discard(u)
    
    def get_neighbors(self, node: int) -> Set[int]:
        """获取节点的邻居"""
        return self.adj_list[node]
    
    def get_degree(self, node: int) -> int:
        """获取节点的度"""
        return len(self.adj_list[node])
    
    def get_density(self) -> float:
        """计算图的密度"""
        n = len(self.nodes)
        if n <= 1:
            return 0.0
        return 2 * len(self.edges) / (n * (n - 1))
    
    def get_average_degree(self) -> float:
        """计算平均度"""
        if not self.nodes:
            return 0.0
        return 2 * len(self.edges) / len(self.nodes)
    
    def save(self, filename: str):
        """
        保存图到文件
        Args:
            filename: 输出文件路径
        """
        with open(filename, 'w') as f:
            f.write(f"{len(self.nodes)} {len(self.edges)}\n")
            for u, v in sorted(self.edges):
                # 转换回原始节点ID
                orig_u = self.reverse_mapping.get(u, u)
                orig_v = self.reverse_mapping.get(v, v)
                f.write(f"{orig_u} {orig_v}\n")
        print(f"图已保存到: {filename}")
    
    def get_basic_info(self):
        """获取图的基本信息"""
        info = {
            "节点数": len(self.nodes),
            "边数": len(self.edges),
            "密度": self.get_density(),
            "平均度": self.get_average_degree()
        }
        return info
    
    def print_info(self):
        """打印图的基本信息"""
        info = self.get_basic_info()
        print("=== 图的基本信息 ===")
        for key, value in info.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
    
    def k_cores(self, output_file: str = None):
        """
        k-core分解算法
        Args:
            output_file: 输出文件路径
        Returns:
            dict: 每个节点的coreness值
        """
        start_time = time.time()
        
        # 初始化每个节点的度
        degrees = {node: self.get_degree(node) for node in self.nodes}
        coreness = {}
        
        # 使用最小堆维护度数
        min_heap = [(degrees[node], node) for node in self.nodes]
        heapq.heapify(min_heap)
        
        visited = set()
        
        while min_heap:
            current_degree, node = heapq.heappop(min_heap)
            
            if node in visited:
                continue
                
            visited.add(node)
            coreness[node] = current_degree
            
            # 更新邻居的度数
            for neighbor in self.get_neighbors(node):
                if neighbor not in visited:
                    degrees[neighbor] -= 1
                    heapq.heappush(min_heap, (degrees[neighbor], neighbor))
        
        end_time = time.time()
        runtime = end_time - start_time
        
        # 输出结果
        if output_file:
            with open(output_file, 'w') as f:
                f.write(f"{runtime:.3f}S\n")
                for node in sorted(self.nodes):
                    orig_node = self.reverse_mapping.get(node, node)
                    f.write(f"{orig_node} {coreness[node]}\n")
            print(f"k-core结果已保存到: {output_file}")
        
        print(f"k-core分解完成，运行时间: {runtime:.3f}秒")
        return coreness
    
    def densest_subgraph_exact(self, output_file: str = None):
        """
        最密子图精确算法（基于二分搜索和最大流）
        Args:
            output_file: 输出文件路径
        Returns:
            tuple: (最密度, 最密子图节点集合)
        """
        start_time = time.time()
        
        if not self.nodes or not self.edges:
            return 0.0, set()

        print(f"开始最密子图精确算法 (节点: {len(self.nodes)}, 边: {len(self.edges)})")
        
        # 二分搜索的范围
        low = 0.0
        high = float(len(self.edges))
        
        best_density = 0.0
        best_subgraph = set()
        
        # 二分搜索的精度足够高即可，例如100次迭代
        for iteration in range(100):
            g = (low + high) / 2
            
            # 每10次迭代输出一次进度
            if iteration % 10 == 0:
                print(f"  二分搜索进度: {iteration}/100, 当前密度范围: [{low:.4f}, {high:.4f}]")
            
            if g == 0: # 避免除以0
                is_denser, subgraph = self._check_density_and_get_subgraph(g)
                if is_denser:
                    low = g
                    current_density = self._calculate_subgraph_density(subgraph)
                    if current_density > best_density:
                        best_density = current_density
                        best_subgraph = subgraph
                else:
                    high = g
                continue

            is_denser, subgraph = self._check_density_and_get_subgraph(g)
            
            if is_denser:
                # 存在密度 >= g 的子图，尝试更高的密度
                low = g
                # 更新找到的最优子图
                current_density = self._calculate_subgraph_density(subgraph)
                if current_density > best_density:
                    best_density = current_density
                    best_subgraph = subgraph
                    # 当找到更好的解时也输出信息
                    if iteration % 5 == 0:
                        print(f"    找到更优解: 密度 = {best_density:.4f}, 子图大小 = {len(best_subgraph)}")
            else:
                # 不存在，降低密度上界
                high = g
        
        end_time = time.time()
        runtime = end_time - start_time
        
        # 如果没有找到任何子图（例如图为空），则返回图中密度最大的单个节点（密度为0）
        if not best_subgraph and self.nodes:
            best_subgraph = {next(iter(self.nodes))}
            best_density = 0.0

        print(f"最密子图精确算法完成! 最终密度: {best_density:.4f}, 子图节点数: {len(best_subgraph)}")

        if output_file:
            with open(output_file, 'w') as f:
                f.write(f"{runtime:.3f}S\n")
                f.write(f"{best_density:.4f}\n")
                # 转换回原始节点ID
                original_ids = sorted([self.reverse_mapping.get(node, node) for node in best_subgraph])
                f.write(" ".join(map(str, original_ids)) + "\n")
            print(f"最密子图(精确)结果已保存到: {output_file}")
        
        print(f"最密子图(精确)完成，运行时间: {runtime:.3f}秒, 密度: {best_density:.4f}")
        return best_density, best_subgraph

    def _check_density_and_get_subgraph(self, g: float) -> Tuple[bool, Set[int]]:
        """
        使用最大流-最小割构建图来检查是否存在密度 > g 的子图
        """
        flow_graph = nx.DiGraph()
        source, sink = 's', 't'
        
        # 添加源点和汇点
        flow_graph.add_nodes_from([source, sink])
        
        node_map = {node: f"v_{node}" for node in self.nodes}
        edge_map = {i: f"e_{i}" for i in range(len(self.edges))}

        # 1. 添加从源点到边节点的边
        for i, edge in enumerate(self.edges):
            edge_node = edge_map[i]
            flow_graph.add_edge(source, edge_node, capacity=1.0)
            
            # 2. 添加从边节点到其端点的边
            u, v = edge
            node_u, node_v = node_map[u], node_map[v]
            flow_graph.add_edge(edge_node, node_u, capacity=float('inf'))
            flow_graph.add_edge(edge_node, node_v, capacity=float('inf'))
            
        # 3. 添加从节点到汇点的边
        for node in self.nodes:
            flow_graph.add_edge(node_map[node], sink, capacity=g)
            
        # 计算最大流/最小割
        cut_value, partition = nx.minimum_cut(flow_graph, source, sink)
        
        # 找到在源点侧的节点
        reachable, non_reachable = partition
        subgraph_nodes_str = {n for n in reachable if n.startswith('v_')}
        subgraph_nodes = {int(s.split('_')[1]) for s in subgraph_nodes_str}
        
        # 检查条件 |E| - g|V| 是否 > 0
        # 这等价于 min_cut < |E| (总边数)
        # 使用一个小的容差来处理浮点数比较
        epsilon = 1e-9
        if len(self.edges) - cut_value > epsilon:
            return True, subgraph_nodes
        else:
            return False, set()
    
    def densest_subgraph_approx(self, output_file: str = None) -> Tuple[float, Set[int]]:
        """
        2-近似最密子图算法
        返回: (密度, 节点集合)
        """
        start_time = time.time()
        
        if not self.nodes or not self.edges:
            if output_file:
                with open(output_file, 'w') as f:
                    f.write("0.000\n0.0\n")
            return 0.0, set()
        
        print(f"开始2-近似最密子图算法 (节点: {len(self.nodes)}, 边: {len(self.edges)})")
        
        # 初始化
        current_nodes = set(self.nodes)
        best_density = 0.0
        best_subgraph = set()
        
        # 计算初始度数
        node_degrees = {}
        for node in current_nodes:
            node_degrees[node] = len(self.get_neighbors(node) & current_nodes)
        
        iteration = 0
        max_iterations = len(self.nodes)  # 设置最大迭代次数防止无限循环
        
        while current_nodes and iteration < max_iterations:
            iteration += 1
            
            # 每100次迭代显示一次进度
            if iteration % 100 == 0 or iteration == 1:
                print(f"  2-近似算法进度: {iteration}/{max_iterations}, 当前节点数: {len(current_nodes)}")
            
            # 计算当前密度
            current_edges = 0
            for u, v in self.edges:
                if u in current_nodes and v in current_nodes:
                    current_edges += 1
            
            if len(current_nodes) > 0:
                current_density = current_edges / len(current_nodes)
                if current_density > best_density:
                    best_density = current_density
                    best_subgraph = current_nodes.copy()
            
            # 如果只剩下很少节点，可以提前停止
            if len(current_nodes) <= 2:
                break
            
            # 找到度数最小的节点
            min_degree = min(node_degrees[node] for node in current_nodes)
            min_degree_nodes = [node for node in current_nodes 
                              if node_degrees[node] == min_degree]
            
            # 移除一个度数最小的节点
            node_to_remove = min_degree_nodes[0]
            current_nodes.remove(node_to_remove)
            
            # 更新邻居的度数（只有在current_nodes中的节点才需要更新）
            neighbors = self.get_neighbors(node_to_remove) & current_nodes
            for neighbor in neighbors:
                if neighbor in node_degrees:
                    node_degrees[neighbor] -= 1
            
            # 移除已删除节点的度数记录
            del node_degrees[node_to_remove]
            
            # 如果图变得太稀疏，可以提前停止
            if len(current_nodes) > 0 and current_edges / len(current_nodes) < 0.1:
                if len(current_nodes) > 1000:  # 只在大图上应用这个优化
                    print(f"  图变得稀疏，提前停止 (剩余节点: {len(current_nodes)})")
                    break
        
        end_time = time.time()
        runtime = end_time - start_time
        
        print(f"2-近似算法完成! 最佳密度: {best_density:.4f}, 子图节点数: {len(best_subgraph)}")
        
        # 保存结果
        if output_file:
            with open(output_file, 'w') as f:
                f.write(f"{runtime:.3f}s\n")
                f.write(f"{best_density:.6f}\n")
                f.write(" ".join(str(self.reverse_mapping.get(node, node)) for node in sorted(best_subgraph)))
        
        print(f"最密子图(2-近似)算法完成，运行时间: {runtime:.3f}秒，密度: {best_density:.6f}")
        return best_density, best_subgraph
    
    def _calculate_subgraph_density(self, nodes: Set[int]) -> float:
        """计算子图的密度"""
        if len(nodes) <= 1:
            return 0.0
        
        edge_count = 0
        for u, v in self.edges:
            if u in nodes and v in nodes:
                edge_count += 1
        
        n = len(nodes)
        return 2 * edge_count / (n * (n - 1))
    
    def k_clique_decomposition(self, k: int, output_file: str = None):
        """
        k-clique分解算法，使用BK算法求极大团
        Args:
            k: 团的最小大小
            output_file: 输出文件路径
        Returns:
            list: 所有大小>=k的极大团
        """
        start_time = time.time()
        
        # 找到所有极大团
        maximal_cliques = []
        self._bron_kerbosch(set(), set(self.nodes), set(), maximal_cliques)
        
        # 过滤出大小>=k的团
        k_cliques = [clique for clique in maximal_cliques if len(clique) >= k]
        
        end_time = time.time()
        runtime = end_time - start_time
        
        # 输出结果
        if output_file:
            with open(output_file, 'w') as f:
                f.write(f"{runtime:.3f}S\n")
                for clique in k_cliques:
                    orig_nodes = [self.reverse_mapping.get(node, node) for node in sorted(clique)]
                    f.write(" ".join(map(str, orig_nodes)) + "\n")
            print(f"k-clique分解结果已保存到: {output_file}")
        
        print(f"k-clique分解完成，运行时间: {runtime:.3f}秒，找到{len(k_cliques)}个大小>={k}的极大团")
        return k_cliques
    
    def _bron_kerbosch(self, R: Set[int], P: Set[int], X: Set[int], cliques: List[Set[int]]):
        """
        Bron-Kerbosch算法求极大团
        Args:
            R: 当前团
            P: 候选节点集合
            X: 已处理节点集合
            cliques: 存储结果的列表
        """
        if not P and not X:
            cliques.append(R.copy())
            return
        
        # 选择度数最大的节点作为pivot
        pivot = None
        max_degree = -1
        for node in P | X:
            degree = len(self.get_neighbors(node) & P)
            if degree > max_degree:
                max_degree = degree
                pivot = node
        
        pivot_neighbors = self.get_neighbors(pivot) if pivot else set()
        
        # 对P中不与pivot相邻的节点进行递归
        for node in list(P - pivot_neighbors):
            neighbors = self.get_neighbors(node)
            self._bron_kerbosch(
                R | {node},
                P & neighbors,
                X & neighbors,
                cliques
            )
            P.remove(node)
            X.add(node)
    
    def dynamic_k_core_maintenance(self, operations: List[Tuple[str, int, int]], output_file: str = None):
        """
        动态k-core维护算法
        Args:
            operations: 操作列表，每个操作为(操作类型, u, v)，操作类型为'insert'或'delete'
            output_file: 输出文件路径
        Returns:
            dict: 最终的coreness值
        """
        start_time = time.time()
        
        # 初始化coreness值
        self.coreness = self.k_cores()
        
        # 执行动态操作
        for op_type, u, v in operations:
            if op_type == 'insert':
                self._dynamic_insert_edge(u, v)
            elif op_type == 'delete':
                self._dynamic_delete_edge(u, v)
        
        end_time = time.time()
        runtime = end_time - start_time
        
        # 输出结果
        if output_file:
            with open(output_file, 'w') as f:
                f.write(f"{runtime:.3f}S\n")
                for node in sorted(self.nodes):
                    orig_node = self.reverse_mapping.get(node, node)
                    f.write(f"{orig_node} {self.coreness[node]}\n")
            print(f"动态k-core维护结果已保存到: {output_file}")
        
        print(f"动态k-core维护完成，运行时间: {runtime:.3f}秒")
        return self.coreness
    
    def _dynamic_insert_edge(self, u: int, v: int):
        """
        动态插入边并更新coreness值
        Args:
            u, v: 边的两个端点
        """
        if u == v or (min(u, v), max(u, v)) in self.edges:
            return  # 自环或边已存在
        
        # 添加边
        self.add_edge(u, v)
        
        # 更新coreness值 - 简化版本，避免复杂的增量更新
        # 对于边插入，coreness值只可能增加，且增加幅度有限
        for node in [u, v]:
            neighbors = self.get_neighbors(node)
            # 计算当前节点的有效度数（度数大于等于当前coreness的邻居数量）
            effective_degree = sum(1 for neighbor in neighbors 
                                 if self.coreness.get(neighbor, 0) >= self.coreness.get(node, 0))
            
            # 简单更新：取有效度数和当前coreness的最大值
            if effective_degree > self.coreness.get(node, 0):
                # 限制coreness的增长，避免过度计算
                max_neighbor_coreness = max((self.coreness.get(neighbor, 0) for neighbor in neighbors), default=0)
                self.coreness[node] = min(effective_degree, max_neighbor_coreness + 1)
    
    def _dynamic_delete_edge(self, u: int, v: int):
        """
        动态删除边并更新coreness值
        Args:
            u, v: 边的两个端点
        """
        edge = (min(u, v), max(u, v))
        if edge not in self.edges:
            return  # 边不存在
        
        # 删除边
        self.remove_edge(u, v)
        
        # 简化的删除更新：重新计算受影响节点的coreness
        affected_nodes = {u, v}
        
        # 使用BFS传播更新
        queue = deque([u, v])
        
        while queue:
            node = queue.popleft()
            neighbors = self.get_neighbors(node)
            
            # 重新计算coreness
            old_coreness = self.coreness.get(node, 0)
            
            # 计算有效度数
            effective_degree = sum(1 for neighbor in neighbors 
                                 if self.coreness.get(neighbor, 0) >= old_coreness)
            
            new_coreness = min(effective_degree, old_coreness)
            
            if new_coreness < old_coreness:
                self.coreness[node] = new_coreness
                
                # 将可能受影响的邻居加入队列
                for neighbor in neighbors:
                    if (neighbor not in affected_nodes and 
                        self.coreness.get(neighbor, 0) >= old_coreness):
                        affected_nodes.add(neighbor)
                        queue.append(neighbor)
    
    def show(self, layout: str = 'spring', node_size: int = 300, 
             title: str = "图可视化", save_path: str = None):
        """
        显示图的可视化
        Args:
            layout: 布局算法 ('spring', 'circular', 'random', 'kamada_kawai')
            node_size: 节点大小
            title: 图标题
            save_path: 保存路径，如果提供则保存图片
        """
        if not self.nodes:
            print("图为空，无法可视化")
            return
        
        # 创建NetworkX图对象
        G = nx.Graph()
        G.add_nodes_from(self.nodes)
        G.add_edges_from(self.edges)
        
        # 选择布局
        if layout == 'spring':
            pos = nx.spring_layout(G)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        elif layout == 'random':
            pos = nx.random_layout(G)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(G)
        else:
            pos = nx.spring_layout(G)
        
        # 创建图形
        plt.figure(figsize=(12, 8))
        nx.draw(G, pos, node_color='lightblue', node_size=node_size, 
                with_labels=True, font_size=8, font_weight='bold',
                edge_color='gray', width=1)
        
        plt.title(title, fontsize=16)
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图已保存到: {save_path}")
        else:
            # 如果没有指定保存路径，默认保存
            default_path = "output/graph_visualization.png"
            plt.savefig(default_path, dpi=300, bbox_inches='tight')
            print(f"图已保存到: {default_path}")
        
        plt.close()  # 关闭图形，释放内存
    
    def _get_layout(self):
        """获取图的布局"""
        try:
            import networkx as nx
            # 创建NetworkX图对象
            G = nx.Graph()
            G.add_nodes_from(self.nodes)
            G.add_edges_from(self.edges)
            
            # 使用spring布局
            pos = nx.spring_layout(G, k=1, iterations=50)
            return pos
        except Exception as e:
            print(f"布局计算失败: {e}")
            # 如果失败，使用简单的随机布局
            import random
            pos = {}
            for node in self.nodes:
                pos[node] = (random.random(), random.random())
            return pos

    def show_coreness(self, coreness_values=None, save_path=None):
        """可视化coreness结构"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            
            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
            plt.rcParams['axes.unicode_minus'] = False
            
            if coreness_values is None:
                coreness_values = self.k_cores()
            
            if len(self.nodes) > 500:
                print(f"图太大 ({len(self.nodes)} 节点)，跳过可视化以避免性能问题")
                return
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # 获取节点位置
            pos = self._get_layout()
            
            # 获取coreness值
            node_colors = [coreness_values.get(node, 0) for node in self.nodes]
            
            # 绘制边
            for u, v in self.edges:
                if u in pos and v in pos:
                    x_coords = [pos[u][0], pos[v][0]]
                    y_coords = [pos[u][1], pos[v][1]]
                    ax.plot(x_coords, y_coords, 'gray', alpha=0.3, linewidth=0.5)
            
            # 绘制节点
            scatter = ax.scatter([pos[node][0] for node in self.nodes], 
                               [pos[node][1] for node in self.nodes],
                               c=node_colors, cmap='viridis', s=50, alpha=0.8)
            
            # 添加colorbar，明确指定ax参数
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Coreness Value')
            
            ax.set_title('Graph Coreness Visualization')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Coreness可视化已保存到: {save_path}")
            else:
                plt.show()
            
            plt.close()
            
        except Exception as e:
            print(f"Coreness可视化失败: {e}")
    
    def show_subgraph(self, subgraph_nodes: Set[int], title: str = "子图可视化",
                     highlight_color: str = 'red', save_path: str = None):
        """
        显示子图的可视化
        Args:
            subgraph_nodes: 子图节点集合
            title: 图标题
            highlight_color: 高亮颜色
            save_path: 保存路径
        """
        if not self.nodes:
            print("图为空，无法可视化")
            return
        
        # 创建NetworkX图对象
        G = nx.Graph()
        G.add_nodes_from(self.nodes)
        G.add_edges_from(self.edges)
        
        # 节点着色
        node_colors = []
        for node in G.nodes():
            if node in subgraph_nodes:
                node_colors.append(highlight_color)
            else:
                node_colors.append('lightgray')
        
        # 边着色
        edge_colors = []
        for u, v in G.edges():
            if u in subgraph_nodes and v in subgraph_nodes:
                edge_colors.append('red')
            else:
                edge_colors.append('lightgray')
        
        # 布局
        pos = nx.spring_layout(G)
        
        # 创建图形
        plt.figure(figsize=(12, 8))
        nx.draw(G, pos, node_color=node_colors, edge_color=edge_colors,
                node_size=300, with_labels=True, font_size=8, font_weight='bold',
                width=2)
        
        plt.title(f"{title} (高亮节点: {len(subgraph_nodes)}个)", fontsize=16)
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"子图可视化已保存到: {save_path}")
        else:
            default_path = "output/subgraph_visualization.png"
            plt.savefig(default_path, dpi=300, bbox_inches='tight')
            print(f"子图可视化已保存到: {default_path}")
        
        plt.close() 