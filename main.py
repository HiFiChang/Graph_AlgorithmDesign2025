from graph import Graph
import sys
import os
import random

def extract_small_dataset(source_file, output_file, target_nodes=2000):
    """
    从大数据集中抽取小的密集子图
    Args:
        source_file: 源数据集文件
        output_file: 输出的小数据集文件
        target_nodes: 目标节点数
    """
    print(f"从 {source_file} 抽取小数据集...")
    
    # 先读取原图
    g = Graph(source_file)
    print(f"原图: {len(g.nodes)} 节点, {len(g.edges)} 条边")
    
    if len(g.nodes) <= target_nodes:
        print("原图已经足够小，直接使用")
        g.save(output_file)
        return output_file
    
    # 策略：选择度数最高的节点作为种子，然后扩展邻居
    node_degrees = [(node, g.get_degree(node)) for node in g.nodes]
    node_degrees.sort(key=lambda x: x[1], reverse=True)
    
    selected_nodes = set()
    
    # 从度数最高的节点开始，逐步添加邻居
    for node, degree in node_degrees:
        if len(selected_nodes) >= target_nodes:
            break
            
        if node not in selected_nodes:
            selected_nodes.add(node)
            # 添加一些邻居节点
            neighbors = list(g.get_neighbors(node))
            # 按邻居的度数排序，优先选择度数高的
            neighbors.sort(key=lambda x: g.get_degree(x), reverse=True)
            
            for neighbor in neighbors:
                if len(selected_nodes) >= target_nodes:
                    break
                selected_nodes.add(neighbor)
    
    # 构建子图
    subgraph = Graph()
    subgraph.node_mapping = {node: i for i, node in enumerate(sorted(selected_nodes))}
    subgraph.reverse_mapping = {i: node for node, i in subgraph.node_mapping.items()}
    
    # 添加节点
    for node in selected_nodes:
        mapped_node = subgraph.node_mapping[node]
        subgraph.add_node(mapped_node)
    
    # 添加边
    for u, v in g.edges:
        if u in selected_nodes and v in selected_nodes:
            mapped_u = subgraph.node_mapping[u]
            mapped_v = subgraph.node_mapping[v]
            subgraph.add_edge(mapped_u, mapped_v)
    
    # 保存子图
    subgraph.save(output_file)
    print(f"抽取完成: {len(subgraph.nodes)} 节点, {len(subgraph.edges)} 条边")
    print(f"子图密度: {subgraph.get_density():.4f}")
    
    return output_file

def demonstrate_dynamic_kcore():
    """专门用于演示动态k-core维护功能"""
    print("\n" + "=" * 20 + " 动态k-core维护演示 " + "="*20)
    g = Graph()
    # 一个简单的图用于演示
    edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4), (4, 5), (4, 6), (5, 6)]
    for u, v in edges:
        g.add_edge(u, v)
    
    print("用于演示的初始图信息:")
    g.print_info()
    initial_coreness = g.k_cores()
    print("初始 Coreness: ", {g.reverse_mapping.get(k, k): v for k, v in initial_coreness.items()})

    # 创建操作序列
    operations = [
        ('delete', 0, 1),  # 删除边(0,1)
        ('insert', 0, 4),  # 插入边(0,4)
    ]
    print("\n执行操作序列:", operations)
    dynamic_coreness = g.dynamic_k_core_maintenance(operations, "output/dynamic_kcore_demo.txt")
    print("动态维护后的 Coreness 值:", {g.reverse_mapping.get(k, k): v for k, v in dynamic_coreness.items()})
    print("=" * 62)

def test_real_dataset(filename, dataset_name):
    """测试真实数据集"""
    print(f"\n=== 处理数据集: {dataset_name} ===")
    
    try:
        # 读取图
        g = Graph(filename)
        g.print_info()
        
        # 保存处理后的图
        output_graph = f"output/{dataset_name}_processed.txt"
        g.save(output_graph)
        
        # k-core分解
        print(f"\n--- {dataset_name} k-core分解 ---")
        coreness = g.k_cores(f"output/{dataset_name}_kcore.txt")
        
        # 最密子图
        print(f"\n--- {dataset_name} 最密子图算法 ---")
        
        # 由于精确算法在大图上非常耗时，可以根据需要选择性运行
        if len(g.nodes) <= 1000: # 只对很小的图运行精确算法
            print("运行精确算法...")
            density_exact, subgraph_exact = g.densest_subgraph_exact(f"output/{dataset_name}_densest_exact.txt")
            g.show_subgraph(subgraph_exact, title=f"{dataset_name} - Densest Subgraph (Exact)", save_path=f"output/{dataset_name}_densest_exact.png")
        else:
            print(f"图太大 ({len(g.nodes)} 节点)，跳过最密子图精确算法以节省时间。")
            print("注：精确算法已在代码中实现，但对于大图计算时间过长。")

        print("运行2-近似算法...")
        density_approx, subgraph_approx = g.densest_subgraph_approx(f"output/{dataset_name}_densest_approx.txt")
        g.show_subgraph(subgraph_approx, title=f"{dataset_name} - Densest Subgraph (Approx)", save_path=f"output/{dataset_name}_densest_approx.png")

        # 对中等大小的图进行coreness可视化
        if len(g.nodes) <= 1000: # 同样调整coreness可视化的阈值
            print(f"\n--- {dataset_name} Coreness 可视化 ---")
            g.show_coreness(coreness, save_path=f"output/{dataset_name}_coreness.png")
        
        # 对于大图，k-clique分解可能很慢，所以只在小图上运行
        if len(g.nodes) <= 1000:
            print(f"\n--- {dataset_name} k-clique分解 ---")
            cliques = g.k_clique_decomposition(3, f"output/{dataset_name}_cliques.txt")
        else:
            print(f"图太大 ({len(g.nodes)} 节点)，跳过k-clique分解")
        
        return g
        
    except Exception as e:
        print(f"处理数据集 {dataset_name} 时出错: {e}")
        return None

def create_output_directory():
    """创建输出目录"""
    if not os.path.exists("output"):
        os.makedirs("output")
        print("创建输出目录: output/")

def main():
    """主函数"""
    print("算法课期末大作业 - 图处理库")
    print("=" * 50)
    
    # 创建输出目录
    create_output_directory()
    
    # 1. 演示动态k-core功能 (满足选做题要求)
    demonstrate_dynamic_kcore()
    
    # 2. 生成小数据集用于完整测试
    print("\n=== 生成测试数据集 ===")
    
    # 优先从 CondMat 抽取，如果不存在则尝试其他数据集
    source_files = ["CondMat.txt", "Amazon.txt", "Gowalla.txt"]
    small_dataset = None
    
    for source_file in source_files:
        if os.path.exists(source_file):
            small_dataset = extract_small_dataset(source_file, "output/small_dataset.txt", target_nodes=800)
            break
    
    if small_dataset:
        print("\n=== 在小数据集上运行完整算法测试 ===")
        test_real_dataset(small_dataset, "SmallDataset")
    else:
        print("未找到源数据集文件，跳过小数据集生成")
    
    # # 3. 处理真实数据集（跳过耗时算法）
    # print("\n=== 在大数据集上运行部分算法 ===")
    # datasets = [
    #     ("CondMat.txt", "CondMat"),
    #     ("Amazon.txt", "Amazon"), 
    #     ("Gowalla.txt", "Gowalla")
    # ]
    
    # for filename, dataset_name in datasets:
    #     if os.path.exists(filename):
    #         print(f"\n--- 处理 {dataset_name} (仅运行快速算法) ---")
    #         try:
    #             g = Graph(filename)
    #             g.print_info()
                
    #             # 只运行k-core
    #             print("运行 k-core 分解...")
    #             g.k_cores(f"output/{dataset_name}_kcore.txt")
                
    #             print("运行最密子图2-近似算法...")
    #             # density_approx, subgraph_approx = g.densest_subgraph_approx(f"output/{dataset_name}_densest_approx.txt")
                
    #             print(f"{dataset_name} 处理完成")
                
    #         except Exception as e:
    #             print(f"处理 {dataset_name} 时出错: {e}")
    #     else:
    #         print(f"数据集文件不存在: {filename}")
    
    print("\n" + "=" * 50)
    print("实验完成！")
    
if __name__ == "__main__":
    main() 