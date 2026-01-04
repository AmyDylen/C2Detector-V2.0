import os
import re
def extract_ports_from_filename(filename):
    """从文件名中智能提取端口号"""
    """精准匹配横线分隔的IPv4/IPv6地址"""
    pattern = r"""
        _                        # 固定分隔符
        (?:                      # IP地址部分
            (?:[a-fA-F0-9-]+)    # 精准匹配横线分隔的IPv6（如 fe80--b91c...）
            _(\d+)               # 第一个端口号
        )
        .*?                      # 中间任意字符
        (?:                      # 第二个IP地址部分
            (?:[a-fA-F0-9-]+)
            _(\d+)               # 第二个端口号
        )
        (?=\.pcap|$)             # 确保以.pcap或字符串结尾
    """
    matches = re.findall(pattern, filename, re.VERBOSE)
    
    # 合并所有找到的端口号并去重
    ports = set()
    for group in matches:
        ports.update({int(g) for g in group if g.isdigit()})
    return ports
def delete_small_pcaps(root_dir, min_size_kb=1, target_ports=None):
    """
    增强版文件清理工具（文件名端口识别方案）
    :param root_dir: 扫描根目录
    :param min_size_kb: 最小保留大小(KB)
    :param target_ports: 需要过滤的端口集合
    """
    deleted_files = []
    error_files = []
    min_size = min_size_kb * 1024
    ports = set(target_ports or [])
    
    # 预编译正则表达式提升性能
    filename_pattern = re.compile(r".*\.pcap(?:\.|$)", re.IGNORECASE)
    
    for root, _, files in os.walk(root_dir):
        for file in files:
            if not filename_pattern.search(file):
                continue
            
            file_path = os.path.join(root, file)
            file_info = {
                "path": file_path,
                "size": os.path.getsize(file_path),
                "reasons": []
            }
            
            try:
                # 条件1：文件大小检查
                if file_info["size"] < min_size:
                    file_info["reasons"].append(f"大小过滤({file_info['size']/1024:.1f}KB)")
                
                # 条件2：文件名端口检查
                if ports:
                    found_ports = extract_ports_from_filename(file)
                    if found_ports & ports:
                        file_info["reasons"].append(
                            f"端口过滤(命中端口：{found_ports & ports})"
                        )
                
                # 执行删除操作
                if file_info["reasons"]:
                    os.remove(file_path)
                    deleted_files.append(file_info)
                    
            except Exception as e:
                error_files.append({"path": file_path, "error": str(e)})
    
    # 可视化报告生成（保持原有格式）
    print(f"\n{' 扫描报告 ':=^40}")
    print(f"扫描目录: {os.path.abspath(root_dir)}")
    print(f"删除策略: 小于{min_size_kb}KB文件 | 过滤端口 {ports or '无'}")
    print(f"删除文件: {len(deleted_files)} 个 | 错误文件: {len(error_files)} 个")
    
def delete_empty_folders(folder_path):
    """
    递归删除指定文件夹及其子文件夹中的所有空文件夹。

    :param folder_path: 要处理的根文件夹路径
    """
    # 遍历所有子文件夹（包括自身）
    for root, dirs, files in os.walk(folder_path, topdown=False):
        # 检查当前文件夹是否为空
        if len(dirs) == 0 and len(files) == 0:
            # 删除空文件夹
            os.rmdir(root)
            #print(f"已删除空文件夹: {root}")
            
if __name__ == "__main__":
    # 配置参数
    config = {
        "target_dir": r"F:\pcaps-datacon2020\white-out\222",
        "danger_ports": {137, 138, 139, 53, 67, 68, 546, 5353, 1900, 17500}
        #"danger_ports": {53,5353}
    }
    
    # 安全验证
    print(f"! 重要操作确认 !".center(50, '='))
    confirm = input(f"即将扫描目录：{config['target_dir']}\n"
                    f"删除所有小于2KB文件及包含危险端口 {config['danger_ports']} 的文件？(y/n) ")
    
    if confirm.lower() == 'y':
        delete_small_pcaps(
            root_dir=config["target_dir"],
            min_size_kb=2,
            target_ports=config["danger_ports"]
        )
        print("\n操作完成！")
    else:
        print("操作已中止")
        
    delete_empty_folders(config['target_dir'])