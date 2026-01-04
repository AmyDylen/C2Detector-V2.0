import os
import shutil
import subprocess
import sys
    
def process_file(orig_path, output_base, splitcap_path, parent_folder):
    """直接处理原始文件"""
    try:
        base_name = os.path.basename(orig_path)
        file_dir = os.path.dirname(orig_path)
        
        # 创建临时分割输出目录
        split_output = os.path.join(file_dir, f"split_temp_{os.getpid()}")
        os.makedirs(split_output, exist_ok=True)
        
        # 执行SplitCap分割
        subprocess.run(
            [splitcap_path, "-r", orig_path, "-o", split_output],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE
        )
        
        # 处理分割结果
        moved_count = 0
        for split_file in os.listdir(split_output):
            if split_file.endswith(".pcap"):
                # 直接使用原始文件名创建目标目录
                target_dir = os.path.join(
                    output_base,
                    parent_folder,
                    os.path.splitext(base_name)[0]  # 使用原始文件名作为分类目录
                )
                os.makedirs(target_dir, exist_ok=True)
                
                src = os.path.join(split_output, split_file)
                dst = os.path.join(target_dir, split_file)
                
                if os.path.exists(dst):
                    os.remove(dst)
                shutil.move(src, dst)
                moved_count += 1
        
        # 清理临时目录
        shutil.rmtree(split_output, ignore_errors=True)
        #print(f"处理完成: {base_name} => 迁移{moved_count}个会话")
        return True
    except Exception as e:
        print(f"处理失败 {base_name}: {str(e)}")
        shutil.rmtree(split_output, ignore_errors=True)
        return False
    
def process_second_file(orig_path, output_base, splitcap_path, parent_folder):
    """直接处理文件"""
    try:
        base_name = os.path.basename(orig_path)
        file_dir = os.path.dirname(orig_path)
        
        # 创建临时分割输出目录
        split_output = os.path.join(file_dir, f"split_temp_{os.getpid()}")
        os.makedirs(split_output, exist_ok=True)
        
        # 执行SplitCap分割
        subprocess.run(
            [splitcap_path, "-r", orig_path, "-s", "packets", "10000", "-o", split_output],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE
        )
        print(split_output)
        # 处理分割结果
        moved_count = 0
        for split_file in os.listdir(split_output):
            if split_file.endswith(".pcap"):
                # 直接使用原始文件名创建目标目录
                target_dir = os.path.join(
                    output_base,
                    parent_folder,
                    os.path.splitext(base_name)[0]  # 使用原始文件名作为分类目录
                )
                os.makedirs(target_dir, exist_ok=True)
                src = os.path.join(split_output, split_file)
                dst = os.path.join(target_dir, split_file)
                
                if os.path.exists(dst):
                    os.remove(dst)
                shutil.move(src, dst)
                moved_count += 1
        
        # 清理临时目录
        shutil.rmtree(split_output, ignore_errors=True)
        #print(f"处理完成: {base_name} => 迁移{moved_count}个会话")
        return True
    except Exception as e:
        print(f"处理失败 {base_name}: {str(e)}")
        shutil.rmtree(split_output, ignore_errors=True)
        return False


def main():
    config = {
        "input_dir": r"F:\pcaps-datacon2020\white",
        "output_dir": r"F:\pcaps-datacon2020\white-out",
        "splitcap_path": r"F:\SplitCap.exe",
        "parent_folder": "111",
        "second_folder": "222"
    }
    
    try:
        # 第一遍处理：按时间分割原始文件
        file_list = []
        for root, _, files in os.walk(config["input_dir"]):
            for file in files:
                if file.lower().endswith(('.pcap', '.pcapng')):
                    file_list.append(os.path.join(root, file))
        
        success_count = 0
        total_files = len(file_list)
        for idx, file_path in enumerate(file_list):
            print(f"正在处理文件 {idx+1}/{total_files}: {os.path.basename(file_path)}")
            result = process_file(
                file_path,
                config["output_dir"],
                config["splitcap_path"],
                config["parent_folder"]
            )
            if result:
                success_count += 1
        print(f"\n第一遍处理完成！成功处理 {success_count}/{total_files} 个文件")
    except Exception as e:
        print(f"致命错误: {str(e)}")
        sys.exit(1)
    
    try:
        
        # 第二遍处理：按数据包数量分割第一次生成的目录中的文件
        second_split_input_dir = os.path.join(config["output_dir"], config["parent_folder"])
        file_list = []
        for root, _, files in os.walk(second_split_input_dir):
            for file in files:
                if file.lower().endswith(('.pcap', '.pcapng')):
                    file_list.append(os.path.join(root, file))
       
        success_count = 0
        total_files = len(file_list)
       
        for idx, file_path in enumerate(file_list):
            #print(f"正在处理文件 {idx+1}/{total_files}: {os.path.basename(file_path)}")
            result = process_second_file(
                file_path,
                config["output_dir"],
                config["splitcap_path"],
                config["second_folder"]
            )
            if result:
                success_count += 1
            
        print(f"\n第二遍处理完成！成功处理 {success_count}/{total_files} 个文件")
    except Exception as e:
        print(f"致命错误: {str(e)}")
        sys.exit(1)
if __name__ == "__main__":
    main()