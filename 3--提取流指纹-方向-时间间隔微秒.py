import os
import subprocess
import csv
from collections import defaultdict
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd

def run_tshark(tshark_path, pcap_file, temp_csv_file, session_id_mapping, protocol_type):
    if protocol_type == 'tcp':
        command = [
            tshark_path,
            "-r", pcap_file,
            "-T", "fields",
            "-e", "frame.time_relative",
            "-e", "ip.src",
            "-e", "ip.dst",
            "-e", "tcp.srcport",
            "-e", "tcp.dstport",
            "-e", "tcp.len",
            "-e", "frame.protocols",  # 添加协议层次信息
            "-E", "separator=|",
            "-E", "occurrence=f"
        ]
    elif protocol_type == 'udp':
        command = [
            tshark_path,
            "-r", pcap_file,
            "-T", "fields",
            "-e", "frame.time_relative",
            "-e", "ip.src",
            "-e", "ip.dst",
            "-e", "udp.srcport",
            "-e", "udp.dstport",
            "-e", "udp.length",
            "-e", "frame.protocols",  # 添加协议层次信息
            "-E", "separator=|",
            "-E", "occurrence=f"
        ]
    try:
        
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        
        output = result.stdout

        session_start_times = {}
        session_end_times = {}
        packet_counts = defaultdict(int)
        last_timestamps = defaultdict(int)  # 用于存储每个会话的上一个时间戳

        rows = []

        for line in output.strip().split('\n'):
            fields = line.split('|')
            if len(fields) >= 7:
                # 当前数据包的绝对时间戳（毫秒）
                current_timestamp_ms = int(float(fields[0]) * 1000000)
                src_ip = fields[1]
                dst_ip = fields[2]
                src_port = fields[3]
                dst_port = fields[4]
                tcp_len = fields[5]
                protocols = fields[6]  # 协议层次信息

                length = int(tcp_len) if tcp_len.isdigit() else 0

                key = frozenset([(src_ip, dst_ip, src_port, dst_port), (dst_ip, src_ip, dst_port, src_port)])
                if key not in session_id_mapping:
                    session_id_mapping[key] = len(session_id_mapping) + 1

                session_id = session_id_mapping[key]
                packet_counts[session_id] += 1

                if session_id not in session_start_times:
                    session_start_times[session_id] = current_timestamp_ms
                session_end_times[session_id] = current_timestamp_ms

                # 计算 delta_t
                if session_id in last_timestamps:
                    delta_t_ms = current_timestamp_ms - last_timestamps[session_id]
                    if delta_t_ms == 0:
                        delta_t_ms = 1 #最小的间隔定义为1
                else:
                    delta_t_ms = 1  # 第一个数据包的 delta_t 设置为 0

                # 更新上一个时间戳
                last_timestamps[session_id] = current_timestamp_ms

                rows.append([session_id, length, delta_t_ms, '', src_ip, dst_ip, src_port, dst_port, protocols, "packet"])  # 增加协议字段

        with open(temp_csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Session ID", "Length", "Timestamp", "Session Duration", "Source IP", "Destination IP", "Source Port", "Destination Port", "Protocol", "Type", "Packet Count"])  # 增加Protocols列
            for row in rows:
                session_id = row[0]
                session_duration = session_end_times[session_id] - session_start_times[session_id]
                row[3] = session_duration
                row.append(packet_counts[session_id])
                writer.writerow(row)

    except subprocess.CalledProcessError as e:
        print("An error occurred while running tshark:", e)
        print("stderr output:", e.stderr)
    except ValueError as ve:
        print(ve)

def process_sessions(temp_csv_file, output_csv_file, session_id_mapping):
    """
    处理会话数据，将属于同一会话的数据包按方向和时间窗口聚合成“数据块”（block）。
    
    输入 CSV 格式（示例）:
        Session ID,Length,Timestamp,Session Duration,Source IP,Destination IP,Source Port,Destination Port,Protocol,Type,Packet Count
    
    输出 CSV 新增字段:
        Protocols（原 Protocol 字段重命名）, Block Count
    
    关键逻辑:
        - 使用 Client Hello 或端口启发式判断客户端 IP（即“正向”）
        - 正向流量长度为正，反向为负
        - 同一会话中，相同方向且时间间隔 ≤1秒的数据包合并为一个 block
    """
    session_dict = defaultdict(list)
    
    # 提取 Client Hello 方向信息（用于更准确判断客户端）
    client_hello_info = session_id_mapping.get('client_hello_info', {})

    # 第一步：读取原始数据，按会话分组（不修改原始行）
    with open(temp_csv_file, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        headers = next(reader)  # 跳过表头

        for row in reader:
            # 构造无向四元组 key（用于匹配会话）
            key = frozenset([
                (row[4], row[5], row[6], row[7]),
                (row[5], row[4], row[7], row[6])
            ])
            session_id = session_id_mapping.get(key)
            if session_id is None:
                continue  # 跳过无法映射的行

            # 创建新行：替换 Session ID，其余保持不变（避免修改原始 row）
            new_row = [session_id] + row[1:]  # row[0] 原为占位符，现替换为真实 session_id
            session_dict[session_id].append(new_row)

    # 第二步：处理每个会话，生成 blocks
    with open(output_csv_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([
            "Session ID", "Length", "Timestamp", "Session Duration",
            "Source IP", "Destination IP", "Source Port", "Destination Port",
            "Protocols", "Type", "Packet Count", "Block Count"
        ])

        tls_common_ports = {80, 443, 8443, 993, 995, 5223, 8080}

        for session_id, packets in session_dict.items():
            if not packets:
                continue

            # 按时间戳排序（确保顺序正确）
            packets.sort(key=lambda x: int(x[2]))

            first_packet = packets[0]
            src_ip, dst_ip = first_packet[4], first_packet[5]
            try:
                src_port = int(first_packet[6])
                dst_port = int(first_packet[7])
            except ValueError:
                continue  # 端口无效，跳过

            # 构造无向 key 用于查找 Client Hello 信息
            key = frozenset([(src_ip, dst_ip, src_port, dst_port),
                             (dst_ip, src_ip, dst_port, src_port)])

            # 判断正向（客户端）IP
            positive_direction = None
            if key in client_hello_info:
                positive_direction = client_hello_info[key]
            else:
                # 启发式：若目标端口是常见 TLS 端口，则源 IP 是客户端
                if dst_port in tls_common_ports:
                    positive_direction = src_ip
                elif src_port in tls_common_ports:
                    positive_direction = dst_ip
                else:
                    # 非标准端口：假设第一个非零长度包的源 IP 是客户端
                    # （也可根据业务调整）
                    positive_direction = src_ip

            # 过滤掉 length=0 的首包（如 TCP SYN）
            filtered_packets = []
            for p in packets:
                try:
                    length = int(p[1])
                except ValueError:
                    continue
                if length == 0 and p == packets[0]:
                    continue  # 跳过首包且 length=0
                filtered_packets.append(p)

            if not filtered_packets:
                continue

            # 第三步：构建带符号长度的新 packet 表示，并合并连续块
            signed_packets = []
            for p in filtered_packets:
                try:
                    length = int(p[1])
                except ValueError:
                    continue
                src = p[4]
                adjusted_length = length if src == positive_direction else -length
                # 创建新 packet 表示（不修改原始数据）
                signed_p = [
                    p[0],  # Session ID
                    adjusted_length,
                    p[2],  # Timestamp
                    p[3],  # Session Duration（后续会被覆盖）
                    p[4], p[5], p[6], p[7],
                    p[8],  # Protocols
                    "block",
                    len(filtered_packets),  # Packet Count（暂存，最终统一写）
                    0  # Block Count 占位
                ]
                signed_packets.append(signed_p)

            # 合并逻辑：同方向 + 时间间隔 ≤1秒（1,000,000 微秒）
            merged_blocks = []
            current_block = signed_packets[0]

            for i in range(1, len(signed_packets)):
                prev = current_block
                curr = signed_packets[i]

                # 检查是否同方向（源 IP 相同）
                same_direction = (prev[4] == curr[4])
                # 检查时间间隔
                time_gap = int(curr[2]) - int(prev[2])
                if same_direction and time_gap <= 1_000_000:
                    # 合并：更新 timestamp 为当前包时间，累加长度
                    current_block[1] += curr[1]
                    current_block[2] = curr[2]  # 使用最新时间戳
                else:
                    merged_blocks.append(current_block)
                    current_block = curr

            merged_blocks.append(current_block)
            block_count = len(merged_blocks)
            session_duration = int(merged_blocks[-1][2]) - int(merged_blocks[0][2])

            # 写入输出文件
            for block in merged_blocks:
                block[3] = session_duration  # 更新 Session Duration
                block[10] = len(filtered_packets)  # Packet Count
                block[11] = block_count        # Block Count
                writer.writerow(block[:12])    # 只写前12列（避免多余字段）
def filter_pcap_file(tshark_path, file_path, output_file, protocol_type):
    
    if protocol_type == 'tcp':
        tshark_command = [
            tshark_path,
            '-r', file_path,
            '-Y', 'tcp && !(tcp.len == 0 || tcp.analysis.retransmission || tcp.analysis.window_update || tcp.analysis.zero_window || tcp.analysis.keep_alive || \
                tcp.port == 137 || tcp.port == 138 || ssdp || dhcp || dns || tls.handshake || nbns )',
            '-w', output_file
        ]
    elif protocol_type == 'udp':
        tshark_command = [
            tshark_path,
            '-r', file_path,
            '-Y', 'udp && !(udp.length == 0 || udp.port == 137 || udp.port == 138 || \
            ssdp || dhcp || dns || nbns || browser || arp || lldp || cldap|| icmp || mdns || stp || eapol || ntp || rtcp)',
            '-w', output_file
        ]
        
    try:
        subprocess.run(tshark_command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error filtering {file_path}: {e}")
        raise

def process_pcap_file(tshark_path, pcap_file, temp_folder, output_folder):
    base_name = os.path.splitext(os.path.basename(pcap_file))[0]
    filtered_pcap_file = os.path.join(temp_folder, f"{base_name}_filtered.pcap")
    temp_csv_file = os.path.join(temp_folder, f"{base_name}_temp.csv")
    output_csv_file = os.path.join(output_folder, f"{base_name}.csv")
    
    # 获取协议类型
    filename = base_name.lower()
    protocol_type = 'udp' if 'udp' in filename else 'tcp'
    
    session_id_mapping = {}
    filter_pcap_file(tshark_path, pcap_file, filtered_pcap_file, protocol_type)
    run_tshark(tshark_path, filtered_pcap_file, temp_csv_file, session_id_mapping,protocol_type)
    process_sessions(temp_csv_file, output_csv_file, session_id_mapping)

def process_pcap_files(tshark_path, folder_path, output_folder, temp_folder):
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    pcap_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.pcap') or file.endswith('.pcapng'):
                pcap_files.append(os.path.join(root, file))

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_pcap_file, tshark_path, pcap, temp_folder, output_folder): pcap for idx, pcap in enumerate(pcap_files)}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing PCAP files"):
            future.result()


def main(tshark_path, folder_path, output_folder, temp_folder):
    if os.path.exists(temp_folder):
        shutil.rmtree(temp_folder)
    
    os.makedirs(temp_folder)
    process_pcap_files(tshark_path, folder_path, output_folder, temp_folder)

if __name__ == "__main__":
    tshark_path = r"C:\Program Files\Wireshark\tshark.exe"  # 指定 tshark 的路径
    folder_path = r"F:\pcaps-datacon2020\white-out\333"  # 你的文件夹路径
    output_folder = "output"  # 输出CSV文件的文件夹
    temp_folder = "packet-csv-temp-folder"  # 临时文件夹路径
    main(tshark_path, folder_path, output_folder, temp_folder)
