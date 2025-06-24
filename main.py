from graph import Graph
import sys
import os

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
        if len(g.nodes) <= 25000: # 对CondMat运行，对更大的图默认跳过
            print("运行精确算法 (可能非常耗时)...")
            density_exact, subgraph_exact = g.densest_subgraph_exact(f"output/{dataset_name}_densest_exact.txt")
            g.show_subgraph(subgraph_exact, title=f"{dataset_name} - Densest Subgraph (Exact)", save_path=f"output/{dataset_name}_densest_exact.png")
        else:
            print(f"图太大 ({len(g.nodes)} 节点)，跳过最密子图精确算法以节省时间。")

        print("运行2-近似算法...")
        density_approx, subgraph_approx = g.densest_subgraph_approx(f"output/{dataset_name}_densest_approx.txt")
        g.show_subgraph(subgraph_approx, title=f"{dataset_name} - Densest Subgraph (Approx)", save_path=f"output/{dataset_name}_densest_approx.png")

        # 对中等大小的图进行coreness可视化
        if len(g.nodes) <= 25000:
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
    
    # 2. 处理真实数据集
    print("\n=== 开始处理真实数据集 ===")
    datasets = [
        ("CondMat.txt", "CondMat"),
        ("Amazon.txt", "Amazon"),
        ("Gowalla.txt", "Gowalla")
    ]
    
    for filename, dataset_name in datasets:
        if os.path.exists(filename):
            test_real_dataset(filename, dataset_name)
        else:
            print(f"数据集文件不存在: {filename}")
    
    print("\n" + "=" * 50)
    print("实验完成！请查看output目录中的结果文件。")

if __name__ == "__main__":
    main() 