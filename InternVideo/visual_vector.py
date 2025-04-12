import numpy as np
import matplotlib
matplotlib.interactive(False)
import matplotlib.pyplot as plt
import hypertools as hyp
import pickle

# 假设vector是一个numpy数组，其中每一行是一个向量
# vector = np.array([...])

def visualize_vectors(vector, hue):
    # 定义降维模型
    # models = ['PCA', 'TSNE', 'UMAP']
    models = ['TSNE']
    # 对每个模型进行降维和可视化
    for i, model in enumerate(models):
        plt.tight_layout()
        hyp.plot(vector, '+', reduce=model, colors=["#0000FF","#FF0000","#008000","#FFFF00","#00FFFF"], ndims=2, save_path="/data/gaohaowen/workspace/"+model+"_time.pdf", hue=hue,legend=["AI","REAL","AI-debias","REAL-debias","p"])

def concatenate_strings(string_list):
    # 使用空字符串作为分隔符将列表中的所有字符串连接起来
    return ''.join(string_list)

def read_list_from_txt(filename):
    with open(filename, 'r') as file:
        group = []  # 用于存储每组154行的数据
        all_groups = []  # 用于存储所有组的数据
        all_tensor_string_list = []
        all_tensor_list = []
        for line in file:
            clean_line = line.strip()
            if clean_line:  # 确保不添加空行
                group.append(clean_line)
            if clean_line.endswith('device=\'cuda:0\')'):
                all_groups.append(group)
                group = []
        
        for list_str in all_groups:
            list_str[0] = list_str[0].replace('tensor([', '')
            list_str[-2] = list_str[-2].replace('],','')
            list_str[-1] = list_str[-1].replace('], device=\'cuda:0\')','')
            list_str[-1] = list_str[-1].replace('device=\'cuda:0\')','')
            all_tensor_string_list.append(concatenate_strings(list_str))

        for tensor_string in all_tensor_string_list:
            tensor_list = [float(num.strip()) for num in tensor_string.strip().split(',')]
            tensor = np.array(tensor_list)
            all_tensor_list.append(tensor)
    return all_tensor_list


def create_array(n):
    # 计算每个数字需要重复的次数
    repeat_times = 1000
    
    # 创建一个空列表来存储数组的元素
    array_elements = []
    
    # 循环n次，每次添加repeat_times个相同的数字到列表中
    for i in range(1, n + 1):
        array_elements.extend([i] * repeat_times)
    
    # 将列表转换为NumPy数组
    return np.array(array_elements)

# 加载.pkl文件中的数组
with open('/data/gaohaowen/workspace/InternVideo-main/InternVideo1/vector_debias_sim_time/p_gen.pkl', 'rb') as file:
    p = pickle.load(file)

with open("/data/gaohaowen/workspace/InternVideo-main/InternVideo1/vector_debias_sim_time/gen_vector.pkl", 'rb') as file:
    gen_vector = pickle.load(file)

with open("/data/gaohaowen/workspace/InternVideo-main/InternVideo1/vector_debias_sim_time/org_vector.pkl", 'rb') as file:
    org_vector = pickle.load(file)

with open("/data/gaohaowen/workspace/InternVideo-main/InternVideo1/vector_debias_sim_time/vector_caption.pkl", 'rb') as file:
    vector_caption = pickle.load(file)

with open("/data/gaohaowen/workspace/InternVideo-main/InternVideo1/vector_debias_sim_time/p_debias_vector.pkl", 'rb') as file:
    p_debias_vector = pickle.load(file)

with open("/data/gaohaowen/workspace/InternVideo-main/InternVideo1/vector_debias_sim_time/p_debias_vector_gen.pkl", 'rb') as file:
    p_debias_vector_gen = pickle.load(file)

p = np.array(p)
gen_vector = np.array(gen_vector)
org_vector = np.array(org_vector)
p_debias_vector = np.array(p_debias_vector)
p_debias_vector_gen = np.array(p_debias_vector_gen)
hue = create_array(5)
print(hue)
# loaded_array = np.concatenate((gen_vector,p,p_debias_vector,org_vector))
loaded_array = np.concatenate((gen_vector,org_vector,p_debias_vector_gen,p_debias_vector,p))
#org和gen是分开的 org cap，gen和cap是一起的
visualize_vectors(loaded_array, hue)