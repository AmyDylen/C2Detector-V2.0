import os
import pandas as pd
def process_directory(input_dir, output_dir):
    """
    递归处理输入目录中的所有CSV文件，按一级子目录分类保存
    """
    # 定义边界值
    SIZE_MIN, SIZE_MAX = 1, 1000000
    TIME_MIN, TIME_MAX = 1, 60000000
    # 遍历输入目录
    for root, dirs, files in os.walk(input_dir):
        # 处理当前目录下的CSV文件
        for file in files:
            if file.lower().endswith('.csv'):
                # 计算分类目录（输入目录的第一层子目录）
                rel_path = os.path.relpath(root, input_dir)
                category = rel_path.split(os.sep)[0] if os.sep in rel_path else rel_path
                
                # 创建分类输出目录
                output_category_dir = os.path.join(output_dir, category)
                os.makedirs(output_category_dir, exist_ok=True)
                
                # 构建输入输出路径
                input_path = os.path.join(root, file)
                output_path = os.path.join(output_category_dir, file)
                
                # 处理单个文件
                process_single_file(input_path, output_path,
                                  SIZE_MIN, SIZE_MAX,
                                  TIME_MIN, TIME_MAX)

def process_single_file(input_path, output_path, 
                      SIZE_MIN, SIZE_MAX, 
                      TIME_MIN, TIME_MAX):
    """
    处理单个CSV文件的核心逻辑
    """
    # 读取原始数据
    df = pd.read_csv(input_path)
    
    state_transitions = []
    
    for session_id, group in df.groupby('Session ID'):
        # 直接使用原始顺序（不进行排序）
        session_group = group
        
        # 遍历连续数据块对
        for i in range(len(session_group)-1):
            prev = session_group.iloc[i]
            curr = session_group.iloc[i+1]
            
            # 处理前一个数据块
            u1_dir = -1 if prev['Length'] < 0 else 1
            u1_size = max(SIZE_MIN, min(abs(prev['Length']), SIZE_MAX))
            u1_time = max(TIME_MIN, min(prev['Timestamp'], TIME_MAX))
            
            # 处理当前数据块
            u2_dir = -1 if curr['Length'] < 0 else 1
            u2_size = max(SIZE_MIN, min(abs(curr['Length']), SIZE_MAX))
            u2_time = max(TIME_MIN, min(curr['Timestamp'], TIME_MAX))
            
            # 计算组合属性
            flag = u1_dir * u2_dir
            ratio = max(0.000001, min(round(u1_size / u2_size, 6), 1000000)) 
            diff = max(0.000001, min(round(u1_time / u2_time, 6), 6000000))
            
            state_transitions.append({
                'Session ID': session_id,
                'direction1': u1_dir,
                'size1': u1_size,
                'time1': u1_time,
                'direction2': u2_dir,
                'size2': u2_size,
                'time2': u2_time,
                'flag': flag,
                'ratio': ratio,
                'diff': diff
            })
    
    # 创建DataFrame并保存
    if state_transitions:
        state_df = pd.DataFrame(state_transitions)
        state_df.to_csv(output_path, index=False)
        #print(f"文件已生成：{output_path}")
    else:
        print(f"警告：{input_path} 无有效数据")
# 使用示例
if __name__ == "__main__":
    input_folder = r"C:\Users\dashu\Desktop\C2Detector-long\output"    # 原始数据根目录
    output_folder = r"C:\Users\dashu\Desktop\C2Detector-long\processed"  # 处理结果根目录
    
    # 清空并创建输出目录
    if os.path.exists(output_folder):
        import shutil
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)
    
    # 开始处理
    process_directory(input_folder, output_folder)
