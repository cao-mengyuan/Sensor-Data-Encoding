import heapq
from collections import defaultdict, Counter
import math
import reedsolo
import numpy as np

def generate_normal_diff_sequence(length, mu, sigma, min_val=-100, max_val=100):
    """
    生成符合正态分布差值的序列
    
    参数:
        length: 序列长度 (默认100)
        mu: 差值的均值 (默认0)
        sigma: 差值的标准差 (默认10)
        min_val: 数据最小值 (默认-100)
        max_val: 数据最大值 (默认100)
    
    返回:
        np.array: 生成的数据序列
    """
    # 初始化序列
    sequence = np.zeros(length)
    sequence[0] = 0  # 第一个数据点为0
    
    # 生成正态分布差值
    diffs = np.random.normal(mu, sigma, length-1)
    
    # 计算累积和并限制范围
    for i in range(1, length):
        sequence[i] = sequence[i-1] + diffs[i-1]
        # 强制限制在[min_val, max_val]范围内
        if sequence[i] > max_val:
            sequence[i] = max_val
        elif sequence[i] < min_val:
            sequence[i] = min_val
    
    return sequence


# 差分编码
def differential_encoding(data):
    diff_data = [data[0]]
    # diff_data = []
    for i in range(1, len(data)):
        diff_data.append(data[i] - data[i - 1])
    return diff_data


# 量化
def quantize(data, min, max, levels):
    """
    将数据量化到指定的离散水平。
    :param data: 差分数据
    :param levels: 量化水平数量
    :return: 量化后的数据
    """
    step = (max - min) / levels
    quantized_data = [round(value / step) * step for value in data]
    return quantized_data

# 计算绝对值之和
def calculate_absolute_error(original_data, decoded_data):
    absolute_error = sum(abs(original - decoded) for original, decoded in zip(original_data, decoded_data))
    return absolute_error

# 差分解码
def differential_decoding(diff_data):
    data = [diff_data[0]]
    for i in range(1, len(diff_data)):
        data.append(data[-1] + diff_data[i])
    return data

# 哈夫曼编码
class HuffmanNode:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(frequencies):
    heap = [HuffmanNode(char, freq) for char, freq in frequencies.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = HuffmanNode(None, left.freq + right.freq)
        merged.left = left
        merged.right = right
        heapq.heappush(heap, merged)
    return heap[0]

def generate_huffman_codes(node, prefix="", code_dict=None):
    if code_dict is None:
        code_dict = {}
    if node is not None:
        if node.char is not None:
            code_dict[node.char] = prefix
        generate_huffman_codes(node.left, prefix + "0", code_dict)
        generate_huffman_codes(node.right, prefix + "1", code_dict)
    return code_dict

def huffman_encode(data, code_dict):
    return ''.join(code_dict[char] for char in data)

def huffman_decode(encoded_data, huffman_tree):
    decoded_data = []
    node = huffman_tree
    for bit in encoded_data:
        if bit == '0':
            node = node.left
        else:
            node = node.right
        if node.char is not None:
            decoded_data.append(node.char)
            node = huffman_tree
    return decoded_data

# 主函数
def differential_quantized_huffman_encode(data, levels):
    # 差分编码
    diff_data = differential_encoding(data)
    # print("差分编码结果：", diff_data)
    
    # 量化
    quantized_data = quantize(diff_data, -3, 3, levels)
    # print("量化后的差分数据：", quantized_data)
    
    # 计算频率
    frequencies = Counter(quantized_data)
    # print("频率统计：", frequencies)
    
    # 构建哈夫曼树
    huffman_tree = build_huffman_tree(frequencies)
    
    # 生成哈夫曼编码表
    huffman_codes = generate_huffman_codes(huffman_tree)
    # print("哈夫曼编码表：", huffman_codes)
    
    # 哈夫曼编码
    encoded_data = huffman_encode(quantized_data, huffman_codes)
    # print("哈夫曼编码结果：", encoded_data)
    
    return encoded_data, huffman_tree

def differential_quantized_huffman_decode(encoded_data, huffman_tree):
    # 哈夫曼解码
    quantized_data = huffman_decode(encoded_data, huffman_tree)
    # print("哈夫曼解码结果（量化后的差分数据）：", quantized_data)
    
    # 差分解码
    original_data = differential_decoding(quantized_data)
    # print("差分解码结果（原始数据）：", original_data)
    
    return original_data

# 二进制序列转换为字节序列
def binary_to_bytes(binary_string):
    return int(binary_string, 2).to_bytes((len(binary_string) + 7) // 8, byteorder='big')

# 字节序列转换为二进制字符串
def bytes_to_binary(byte_array):
    return ''.join(format(byte, '08b') for byte in byte_array)

# RS码
def rs_encode(binary_bytes, n, k):
    rs = reedsolo.RSCodec(n - k)
    encoded = rs.encode(binary_bytes)
    return encoded

def flip_bit_in_string(binary_string, position):
    """
    翻转字符串中指定位置的位。
    :param binary_string: 二进制字符串（只包含 '0' 和 '1'）
    :param position: 要翻转的位的位置（从0开始计数）
    :return: 修改后的二进制字符串
    """
    # 确保位置在有效范围内
    if position < 0 or position >= len(binary_string):
        raise ValueError("位置超出字符串的范围")
    
    # 将字符串转换为列表以便修改
    binary_list = list(binary_string)
    
    # 翻转指定位置的位
    if binary_list[position] == '0':
        binary_list[position] = '1'
    else:
        binary_list[position] = '0'
    
    # 将列表重新转换为字符串
    modified_string = ''.join(binary_list)
    
    return modified_string


snr_list = np.linspace(1.5, 5, num=int((5-1.5)/0.1)+1)
cnt_list = []
T = 2000 # 实验次数

for SNR in snr_list:
    cnt = 0 # 能成功解码的概率
    for i in range(T):
        # 生成数据
        temperature_data = generate_normal_diff_sequence(length=100, mu=0, sigma=5)

        # 编码
        encoded_data, huffman_tree = differential_quantized_huffman_encode(temperature_data, levels = 256)

        # print(len(encoded_data))

        # 使用RS码进行编码
        n = 255  # 码字长度
        k = 223  # 数据长度
        RS_encoded_message = rs_encode(binary_to_bytes(encoded_data), n, k)

        RS_encoded_binary = bytes_to_binary(RS_encoded_message)
        # print("Encoded message:", RS_encoded_binary)

        # print(len(RS_encoded_binary))
        #------------------------------------------------------------
        # 调制成BPSK
        data = np.array([int(bit) for bit in RS_encoded_binary])
        bpskSignal = 2 * data - 1
        # 计算噪声功率
        noisePower = 1 / (10 ** (SNR/10))

        # 生成高斯白噪声
        noise = np.sqrt(noisePower / 2) * (np.random.randn(len(data)) + 1j * np.random.randn(len(data)))

        # 添加噪声到信号
        noisy_signal = bpskSignal + noise

        # BPSK解调
        # 将接收到的信号与0比较，如果大于0，则解调为1，否则解调为0
        demodulated_data = (np.real(noisy_signal) > 0).astype(int)

        # 将解调后的数字列表转换回二进制序列字符串
        demodulated_sequence = ''.join(str(bit) for bit in demodulated_data)

        # -----------------------------------RS 解码------------------
        #先转换为bytearray形式
        bytes_msg=bytes(int(demodulated_sequence[i:i+8],2) for i in range(0,len(demodulated_sequence),8))
        array_msg=bytearray(bytes_msg)
        #再纠错
        ecc = reedsolo.RSCodec(n - k)

        try:
            data, temp1, temp2 = ecc.decode(array_msg)
            binary_code=''.join(format(x,'08b') for x in data)
            print(i,":解码成功")
        except reedsolo.ReedSolomonError:
            # 解码失败时返回全零或其他约定标识
            binary_code = None # 返回k字节的全零
            print(i,":解码失败，返回默认值")
        except Exception as e:
            # 捕获其他意外错误
            binary_code = None
            print(f"发生未知错误: {str(e)}")

        if binary_code is not None:
            cnt = cnt + 1
            pass
        else:
            # 错误处理逻辑
            pass

    print(cnt / T)
    cnt_list.append(cnt/T)

print(cnt_list)
# 解码
# decoded_data = differential_quantized_huffman_decode(binary_code, huffman_tree)

# # 计算绝对值之和
# absolute_error = calculate_absolute_error(temperature_data, decoded_data)
# print("平均绝对值误差：", absolute_error/len(temperature_data))
    
    