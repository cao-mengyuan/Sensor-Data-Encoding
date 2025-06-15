import heapq
from collections import defaultdict, Counter
import math
import reedsolo

# 定义温度数据数组
# 例如：一天内每小时的温度变化（假设是室内温度）
# temperature_data = [
#     22.0, 21.8, 21.5, 21.2, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0,
#     29.0, 29.5, 29.0, 28.0, 27.0, 26.0, 25.0, 24.0, 23.0, 22.0, 21.5, 21.0
# ]
# temperature_data = [
#     1,2,3,0,-1,-2,-1,0,1,2,3,0,-1,-2,-1,0,-1
# ]
# def read_text_file(file_path):
#     with open(file_path, 'r', encoding='utf-8') as file:
#         text = file.read()
#     return text

# temperature_data = [
#     -2.1, -2.0, -1.9, -1.8, -1.7, -1.6, -1.5, -1.4, -1.3, -1.2, -1.3, -1.2, -1.1, -1# 第1天夜晚
# ]
# 示例：读取文件
# file_path = 'temp.txt'  # 替换为你的文件路径
# temperature_data = read_text_file(file_path)
# 适用于每行一个数据的格式
with open('temp.txt') as f:
    temperature_data = [float(x) for line in f for x in line.strip().split(', ')]
print("确定性的温度传感器数据数组：")
print(temperature_data)

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
    print("差分编码结果：", diff_data)
    
    # 量化
    quantized_data = quantize(diff_data, -3, 3, levels)
    print("量化后的差分数据：", quantized_data)
    
    # 计算频率
    frequencies = Counter(quantized_data)
    print("频率统计：", frequencies)
    
    # 构建哈夫曼树
    huffman_tree = build_huffman_tree(frequencies)
    
    # 生成哈夫曼编码表
    huffman_codes = generate_huffman_codes(huffman_tree)
    print("哈夫曼编码表：", huffman_codes)
    
    # 哈夫曼编码
    encoded_data = huffman_encode(quantized_data, huffman_codes)
    print("哈夫曼编码结果：", encoded_data)
    
    return encoded_data, huffman_tree

def differential_quantized_huffman_decode(encoded_data, huffman_tree):
    # 哈夫曼解码
    quantized_data = huffman_decode(encoded_data, huffman_tree)
    print("哈夫曼解码结果（量化后的差分数据）：", quantized_data)
    
    # 差分解码
    original_data = differential_decoding(quantized_data)
    print("差分解码结果（原始数据）：", original_data)
    
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


# diff_data = differential_encoding(temperature_data)
# print("差分编码结果：", diff_data)

# # 量化
# quantized_data = quantize(diff_data, min = -3, max = 3,levels = 256) # 8bit量化
# print("量化后的差分数据：", quantized_data)

# 编码
encoded_data, huffman_tree = differential_quantized_huffman_encode(temperature_data, levels = 512)

print(len(encoded_data))

# 使用RS码进行编码
n = 255  # 码字长度
k = 223  # 数据长度
RS_encoded_message = rs_encode(binary_to_bytes(encoded_data), n, k)

RS_encoded_binary = bytes_to_binary(RS_encoded_message)
print("Encoded message:", RS_encoded_binary)

#------------------------------------------------------------

# RS_encoded_binary = flip_bit_in_string(RS_encoded_binary,4)
# RS_encoded_binary = flip_bit_in_string(RS_encoded_binary,1)
# RS_encoded_binary = flip_bit_in_string(RS_encoded_binary,18)
# RS_encoded_binary = flip_bit_in_string(RS_encoded_binary,6)

# -----------------------------------RS 解码------------------
#先转换为bytearray形式
bytes_msg=bytes(int(RS_encoded_binary[i:i+8],2) for i in range(0,len(RS_encoded_binary),8))
array_msg=bytearray(bytes_msg)
#再纠错
ecc = reedsolo.RSCodec(n - k)
data, temp1, temp2 = ecc.decode(array_msg)
binary_code=''.join(format(x,'08b') for x in data)
print(binary_code)

# 解码
decoded_data = differential_quantized_huffman_decode(binary_code, huffman_tree)


# 计算绝对值之和
absolute_error = calculate_absolute_error(temperature_data, decoded_data)
print("平均绝对值误差：", absolute_error/len(temperature_data))
    
    