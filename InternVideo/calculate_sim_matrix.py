import numpy as np
import scipy
import csv
import pickle


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

def calculate_p(tensor_list_A, tensor_list_B):

    def difference_vector(vector_a, vector_b):
        # 计算两个向量之间的差值向量
        diff_vector = vector_a - vector_b
        return diff_vector

    def mean_vector(vector_list):
        # 将列表转换为NumPy数组
        array = np.array(vector_list)
        # 计算均值向量
        # axis=0 表示沿着第一个轴（行）计算均值，即对每个列的元素取均值
        mean_vec = np.mean(array, axis=0)
        return mean_vec
    
    diff_list = []
    for tensorA, tensorB in zip(tensor_list_A, tensor_list_B):
        dif = difference_vector(tensorA, tensorB)
        diff_list.append(dif)
    mean_dif_vector = mean_vector(diff_list)
    return diff_list, mean_dif_vector


def cosine_similarity(vector_a, vector_b):
   # 计算两个向量的点积
    dot_product = np.dot(vector_a, vector_b)
    
    # 计算两个向量的范数
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    
    # 计算余弦相似度
    if norm_a == 0 or norm_b == 0:
        # 如果任一向量的范数为0，则相似度为0
        return 0
    else:
        cosine_sim = dot_product / (norm_a * norm_b)
        return cosine_sim


def calculate_cosine(tensor_list_A, tensor_list_B):
    cos_sim_list = []
    for tensorA in tensor_list_A:
        for tensorB in tensor_list_B:
            cos_sim_list.append(cosine_similarity(tensorA,tensorB))
    cos_sim_matrix = np.array(cos_sim_list).reshape(len(tensor_list_A),len(tensor_list_A))
    return cos_sim_matrix

def add_p_to_list(tensor_list, p_vector):
    p_vector_list = []
    for tensor in tensor_list:
        p_vector_list.append(tensor - p_vector)
    return p_vector_list

def save_numpy_array(np_array, file_path):
    # 保存NumPy数组到文件
    np.save(file_path, np_array)
    np.savetxt(file_path+".txt", np_array)

def rank_in_rows(matrix):
    """
    将输入的二维矩阵按行求每个元素是这一行的第几大,最大的保存为0。

    参数:
    matrix : np.array
        输入的二维矩阵。

    返回:
    np.array
        排名矩阵，维度与输入矩阵相同。
    """
    # 初始化排名矩阵，填充-1表示尚未计算排名
    rank_matrix = np.full(matrix.shape, -1)

    # 按行遍历矩阵
    for i, row in enumerate(matrix):
        # 对每一行进行排序并获取排名
        sorted_row = np.argsort(row)
        # 计算排名，最大的排名为0
        rank_matrix[i, sorted_row] = np.arange(row.size)[::-1]
    return rank_matrix

# 按照位置与random_number计算融合后的排名 用于计算ΔΔ
def get_rank_new(gen_rank, org_rank,random_number=0):
    """
    gen_RANK
    """
    gen_rank = gen_rank+1
    org_rank = org_rank+1
    if gen_rank.ndim != 1 or org_rank.ndim != 1:
        raise ValueError("gen_rank and org_rank must be 1-dimensional arrays.")
    if gen_rank.shape != org_rank.shape:
        raise ValueError("gen_rank and org_rank must have the same dimensions.")
    if not (gen_rank > 0).all():
        raise ValueError("All elements in gen_rank must be positive integers.")
    if not (org_rank > 0).all():
        raise ValueError("All elements in org_rank must be positive integers.")
    # 初始化新的数组
    gen_rank_new = np.empty_like(gen_rank)
    org_rank_new = np.empty_like(org_rank)
    # 遍历数组
    for i in range(len(gen_rank)):
        # 生成随机整数
        # 根据条件修改元素
        if random_number % 2 == 0:
            gen_rank_new[i] = 2 * gen_rank[i]
            org_rank_new[i] = 2 * org_rank[i] - 1
        else:
            gen_rank_new[i] = 2 * gen_rank[i] - 1
            org_rank_new[i] = 2 * org_rank[i]
    return (gen_rank_new, org_rank_new)

def cols2metrics(cols, num_queries):
    """
    计算给定列向量中元素的排名指标。

    参数:
    cols -- 一个包含排名的NumPy数组。
    num_queries -- 进行排名查询的总数。

    返回:
    一个字典，包含以下指标：
    - R1: 第一个位置（最顶部）的查询的百分比（100% * 在第一位的查询数 / 查询总数）。
    - R3: 前三个位置的查询的百分比。
    - R5: 前五个位置的查询的百分比。
    - R10: 前十个位置的查询的百分比。
    - R50: 前五十个位置的查询的百分比。
    - MedR: 中位数排名加1。
    - MeanR: 平均排名加1。
    - geometric_mean_R1-R5-R10: R1, R5, R10的几何平均值。

    功能:
    该函数计算并返回一个字典，其中包含查询排名的不同指标。
    这些指标通常用于评估搜索系统或推荐系统的性能。
    """
    metrics = {}
    metrics["R1"] = 100 * float(np.sum(cols == 0)) / num_queries
    metrics["R3"] = 100 * float(np.sum(cols < 3)) / num_queries
    metrics["R5"] = 100 * float(np.sum(cols < 5)) / num_queries
    metrics["R10"] = 100 * float(np.sum(cols < 10)) / num_queries
    metrics["R50"] = 100 * float(np.sum(cols < 50)) / num_queries
    metrics["MedR"] = np.median(cols) + 1
    metrics["MeanR"] = np.mean(cols) + 1
    stats = [metrics[x] for x in ("R1", "R5", "R10")]
    metrics["geometric_mean_R1-R5-R10"] = scipy.stats.mstats.gmean(stats)
    return metrics

def average_values(dict1, dict2):
    # 创建一个新的字典来存储结果
    result = {}
    
    # 合并两个字典的键
    all_keys = set(dict1.keys()) | set(dict2.keys())
    
    # 遍历所有键
    for key in all_keys:
        # 如果键在两个字典中都存在，则计算平均值
        if key in dict1 and key in dict2:
            result[key] = (dict1[key] + dict2[key]) / 2
        # 如果键只在一个字典中存在，则直接添加到结果字典中
        elif key in dict1:
            result[key] = dict1[key]
        elif key in dict2:
            result[key] = dict2[key]
    
    return result


gen_vector = read_list_from_txt("/data/gaohaowen/workspace/InternVideo-main/InternVideo1/vector_output/vision_vector_eval_opensora_fusion_gen.txt")

org_vector = read_list_from_txt("/data/gaohaowen/workspace/InternVideo-main/InternVideo1/vector_output/vision_vector_eval_org.txt")

debias_gen_vector = read_list_from_txt("/data/gaohaowen/workspace/InternVideo-main/InternVideo1/vector_output_debias/vision_vector_eval_opensora_fusion_gen.txt")

debias_org_vector = read_list_from_txt("/data/gaohaowen/workspace/InternVideo-main/InternVideo1/vector_output_debias/vision_vector_eval_org.txt")

vector_caption = read_list_from_txt("/data/gaohaowen/workspace/InternVideo-main/InternVideo1/vector_output/caption_vector_eval_org.txt")

debias_vector_caption = read_list_from_txt("/data/gaohaowen/workspace/InternVideo-main/InternVideo1/vector_output_debias/caption_vector_eval_org.txt")

# 计算p向量 = debias - org 得到debias应该多什么
p_list_gen, p_gen = calculate_p(debias_gen_vector, gen_vector)

p_list_org, p_org = calculate_p(debias_org_vector, org_vector)

with open("/data/gaohaowen/workspace/InternVideo-main/InternVideo1/vector_debias_sim/p_gen.pkl", 'wb') as file:
    pickle.dump(p_list_gen, file)

with open("/data/gaohaowen/workspace/InternVideo-main/InternVideo1/vector_debias_sim/p_org.pkl", 'wb') as file:
    pickle.dump(p_list_org, file)

with open("/data/gaohaowen/workspace/InternVideo-main/InternVideo1/vector_debias_sim/gen_vector.pkl", 'wb') as file:
    pickle.dump(gen_vector, file)

with open("/data/gaohaowen/workspace/InternVideo-main/InternVideo1/vector_debias_sim/org_vector.pkl", 'wb') as file:
    pickle.dump(org_vector, file)

with open("/data/gaohaowen/workspace/InternVideo-main/InternVideo1/vector_debias_sim/vector_caption.pkl", 'wb') as file:
    pickle.dump(vector_caption, file)




p_debias_vector = add_p_to_list(org_vector,p_gen)

p_debias_vector_gen = add_p_to_list(gen_vector,p_gen)

org_sim_matrix = calculate_cosine(org_vector,vector_caption)

p_sim_matrix = calculate_cosine(p_debias_vector,debias_vector_caption)

gen_sim_matrix = calculate_cosine(gen_vector,vector_caption)

with open("/data/gaohaowen/workspace/InternVideo-main/InternVideo1/vector_debias_sim/p_debias_vector.pkl", 'wb') as file:
    pickle.dump(p_debias_vector, file)

with open("/data/gaohaowen/workspace/InternVideo-main/InternVideo1/vector_debias_sim/p_debias_vector_gen.pkl", 'wb') as file:
    pickle.dump(p_debias_vector_gen, file)

save_numpy_array(org_sim_matrix,"/data/gaohaowen/workspace/InternVideo-main/InternVideo1/vector_debias_sim/sim_org")

save_numpy_array(p_sim_matrix,"/data/gaohaowen/workspace/InternVideo-main/InternVideo1/vector_debias_sim/sim_p")

save_numpy_array(gen_sim_matrix,"/data/gaohaowen/workspace/InternVideo-main/InternVideo1/vector_debias_sim/sim_gen")

# save_numpy_array(debias_sim_matrix,"/data/gaohaowen/workspace/InternVideo-main/InternVideo1/vector_debias_sim/sim_debias")

rank = rank_in_rows(np.hstack((org_sim_matrix,gen_sim_matrix)))

rank_p_debias = rank_in_rows(np.hstack((p_sim_matrix,gen_sim_matrix)))

rank_org = rank_in_rows(org_sim_matrix)

rank_gen = rank_in_rows(gen_sim_matrix)

rank_org_p_debias = rank_in_rows(p_sim_matrix)

sinlge_size = rank_org.shape[0]


metrics_org_single = cols2metrics(np.diag(rank_org),sinlge_size)
metrics_gen_single = cols2metrics(np.diag(rank_gen),sinlge_size)
metrics_org_single_p = cols2metrics(np.diag(rank_org_p_debias),sinlge_size)
metrics_gen_single_p = cols2metrics(np.diag(rank_gen),sinlge_size)

metrics_org = cols2metrics(np.diag(rank[:,:sinlge_size]),sinlge_size)

metrics_gen = cols2metrics(np.diag(rank[:,sinlge_size:]),sinlge_size)

metrics_org_p = cols2metrics(np.diag(rank_p_debias[:,:sinlge_size]),sinlge_size)

metrics_gen_p = cols2metrics(np.diag(rank_p_debias[:,sinlge_size:]),sinlge_size)

rank_position_gen1, rank_position_org1 = get_rank_new(np.diag(rank_gen),np.diag(rank_org),0)

rank_position_gen2, rank_position_org2 = get_rank_new(np.diag(rank_gen),np.diag(rank_org),1)

rank_position_gen1_p, rank_position_org1_p = get_rank_new(np.diag(rank_gen),np.diag(rank_org_p_debias),0)

rank_position_gen2_p, rank_position_org2_p = get_rank_new(np.diag(rank_gen),np.diag(rank_org_p_debias),1) 

rank_position_gen1 = rank_position_gen1-1
rank_position_org1 = rank_position_org1-1
rank_position_gen2 = rank_position_gen2-1
rank_position_org2 = rank_position_org2-1

rank_position_gen1_p = rank_position_gen1_p-1
rank_position_org1_p = rank_position_org1_p-1
rank_position_gen2_p = rank_position_gen2_p-1
rank_position_org2_p = rank_position_org2_p-1

metrics_position_gen1 = cols2metrics(rank_position_gen1,sinlge_size)
metrics_position_org1 = cols2metrics(rank_position_org1,sinlge_size)
metrics_position_gen2 = cols2metrics(rank_position_gen2,sinlge_size)
metrics_position_org2 = cols2metrics(rank_position_org2,sinlge_size)
metrics_rank_gen_new = average_values(metrics_position_gen1, metrics_position_gen2)
metrics_rank_org_new = average_values(metrics_position_org1, metrics_position_org2)

metrics_position_gen1_p = cols2metrics(rank_position_gen1_p,sinlge_size)
metrics_position_org1_p = cols2metrics(rank_position_org1_p,sinlge_size)
metrics_position_gen2_p = cols2metrics(rank_position_gen2_p,sinlge_size)
metrics_position_org2_p = cols2metrics(rank_position_org2_p,sinlge_size)
metrics_rank_gen_new_p = average_values(metrics_position_gen1_p, metrics_position_gen2_p)
metrics_rank_org_new_p = average_values(metrics_position_org1_p, metrics_position_org2_p)

csv_data = []
csv_data_p = []
csv_data.append(["org_single",metrics_org_single['R1'],metrics_org_single['R5'],metrics_org_single['R10'],metrics_org_single['MedR'],metrics_org_single['MeanR'],0,0])
csv_data.append(["gen_single",metrics_gen_single['R1'],metrics_gen_single['R5'],metrics_gen_single['R10'],metrics_gen_single['MedR'],metrics_gen_single['MeanR'],0,0])
csv_data.append(["org",metrics_org['R1'],metrics_org['R5'],metrics_org['R10'],metrics_org['MedR'],metrics_org['MeanR'],0,0])
csv_data.append(["gen",metrics_gen['R1'],metrics_gen['R5'],metrics_gen['R10'],metrics_gen['MedR'],metrics_gen['MeanR'],0,0])
csv_data.append(["org_new",metrics_rank_org_new['R1'],metrics_rank_org_new['R5'],metrics_rank_org_new['R10'],metrics_rank_org_new['MedR'],metrics_rank_org_new['MeanR'],0,0])
csv_data.append(["gen_new",metrics_rank_gen_new['R1'],metrics_rank_gen_new['R5'],metrics_rank_gen_new['R10'],metrics_rank_gen_new['MedR'],metrics_rank_gen_new['MeanR'],0,0])
delta_R1 = round(200*((metrics_org['R1']-metrics_gen['R1'])/(metrics_org['R1']+metrics_gen['R1'])), 2)
delta_R3 = round(200*((metrics_org['R3']-metrics_gen['R3'])/(metrics_org['R3']+metrics_gen['R3'])), 2)
delta_R5 = round(200*((metrics_org['R5']-metrics_gen['R5'])/(metrics_org['R5']+metrics_gen['R5'])), 2)
delta_R10 = round(200*((metrics_org['R10']-metrics_gen['R10'])/(metrics_org['R10']+metrics_gen['R10'])), 2)
delta_MedR = round(200*((1/metrics_org['MedR']-1/metrics_gen['MedR'])/(1/metrics_org['MedR']+1/metrics_gen['MedR'])), 2)
delta_MeanR = round(200*((1/metrics_org['MeanR']-1/metrics_gen['MeanR'])/(1/metrics_org['MeanR']+1/metrics_gen['MeanR'])), 2)
delta_R1_new = round(200*((metrics_rank_org_new['R1']-metrics_rank_gen_new['R1'])/(metrics_rank_org_new['R1']+metrics_rank_gen_new['R1'])), 2)
delta_R3_new = round(200*((metrics_rank_org_new['R3']-metrics_rank_gen_new['R3'])/(metrics_rank_org_new['R3']+metrics_rank_gen_new['R3'])), 2)
delta_R5_new = round(200*((metrics_rank_org_new['R5']-metrics_rank_gen_new['R5'])/(metrics_rank_org_new['R5']+metrics_rank_gen_new['R5'])), 2)
delta_R10_new = round(200*((metrics_rank_org_new['R10']-metrics_rank_gen_new['R10'])/(metrics_rank_org_new['R10']+metrics_rank_gen_new['R10'])), 2)
delta_MedR_new = round(200*((1/metrics_rank_org_new['MedR']-1/metrics_rank_gen_new['MedR'])/(1/metrics_rank_org_new['MedR']+1/metrics_rank_gen_new['MedR'])), 2)
delta_MeanR_new = round(200*((1/metrics_rank_org_new['MeanR']-1/metrics_rank_gen_new['MeanR'])/(1/metrics_rank_org_new['MeanR']+1/metrics_rank_gen_new['MeanR'])), 2)
csv_data.append(["delta",delta_R1,delta_R5,delta_R10,delta_MedR,delta_MeanR,round(((delta_R1+delta_MeanR)/2), 2),round(((delta_R1+delta_MedR+delta_MeanR)/3), 2)])
csv_data.append(["delta_new",delta_R1_new,delta_R5_new,delta_R10_new,delta_MedR_new,delta_MeanR_new,round((delta_R1_new+delta_MeanR_new)/2,2),round((delta_R1_new+delta_MedR_new+delta_MeanR_new)/3, 2)])
csv_data.append(["ΔΔ",round(-delta_R1_new+delta_R1, 2),
round(-delta_R5_new+delta_R5, 2),
round(-delta_R10_new+delta_R10, 2),
round(-delta_MedR_new+delta_MedR, 2),
round(-delta_MeanR_new+delta_MeanR, 2),
round(-((delta_R1_new+delta_MeanR_new)/2+(-delta_R1-delta_MeanR)/2), 2),
round(-((delta_R1_new+delta_MedR_new+delta_MeanR_new)/3+(-delta_R1-delta_MedR-delta_MeanR))/3, 2)])


csv_data_p.append(["org_single",metrics_org_single_p['R1'],metrics_org_single_p['R5'],metrics_org_single_p['R10'],metrics_org_single_p['MedR'],metrics_org_single_p['MeanR'],0,0])
csv_data_p.append(["gen_single",metrics_gen_single_p['R1'],metrics_gen_single_p['R5'],metrics_gen_single_p['R10'],metrics_gen_single_p['MedR'],metrics_gen_single_p['MeanR'],0,0])
csv_data_p.append(["org",metrics_org_p['R1'],metrics_org_p['R5'],metrics_org_p['R10'],metrics_org_p['MedR'],metrics_org_p['MeanR'],0,0])
csv_data_p.append(["gen",metrics_gen_p['R1'],metrics_gen_p['R5'],metrics_gen_p['R10'],metrics_gen_p['MedR'],metrics_gen_p['MeanR'],0,0])
csv_data_p.append(["org_new",metrics_rank_org_new_p['R1'],metrics_rank_org_new_p['R5'],metrics_rank_org_new_p['R10'],metrics_rank_org_new_p['MedR'],metrics_rank_org_new_p['MeanR'],0,0])
csv_data_p.append(["gen_new",metrics_rank_gen_new_p['R1'],metrics_rank_gen_new_p['R5'],metrics_rank_gen_new_p['R10'],metrics_rank_gen_new_p['MedR'],metrics_rank_gen_new_p['MeanR'],0,0])
delta_R1_p = round(200*((metrics_org_p['R1']-metrics_gen_p['R1'])/(metrics_org_p['R1']+metrics_gen_p['R1'])), 2)
delta_R3_p = round(200*((metrics_org_p['R3']-metrics_gen_p['R3'])/(metrics_org_p['R3']+metrics_gen_p['R3'])), 2)
delta_R5_p = round(200*((metrics_org_p['R5']-metrics_gen_p['R5'])/(metrics_org_p['R5']+metrics_gen_p['R5'])), 2)
delta_R10_p = round(200*((metrics_org_p['R10']-metrics_gen_p['R10'])/(metrics_org_p['R10']+metrics_gen_p['R10'])), 2)
delta_MedR_p = round(200*((1/metrics_org_p['MedR']-1/metrics_gen_p['MedR'])/(1/metrics_org_p['MedR']+1/metrics_gen_p['MedR'])), 2)
delta_MeanR_p = round(200*((1/metrics_org_p['MeanR']-1/metrics_gen_p['MeanR'])/(1/metrics_org_p['MeanR']+1/metrics_gen_p['MeanR'])), 2)
delta_R1_new_p = round(200*((metrics_rank_org_new_p['R1']-metrics_rank_gen_new_p['R1'])/(metrics_rank_org_new_p['R1']+metrics_rank_gen_new_p['R1'])), 2)
delta_R3_new_p = round(200*((metrics_rank_org_new_p['R3']-metrics_rank_gen_new_p['R3'])/(metrics_rank_org_new_p['R3']+metrics_rank_gen_new_p['R3'])), 2)
delta_R5_new_p = round(200*((metrics_rank_org_new_p['R5']-metrics_rank_gen_new_p['R5'])/(metrics_rank_org_new_p['R5']+metrics_rank_gen_new_p['R5'])), 2)
delta_R10_new_p = round(200*((metrics_rank_org_new_p['R10']-metrics_rank_gen_new_p['R10'])/(metrics_rank_org_new_p['R10']+metrics_rank_gen_new_p['R10'])), 2)
delta_MedR_new_p = round(200*((1/metrics_rank_org_new_p['MedR']-1/metrics_rank_gen_new_p['MedR'])/(1/metrics_rank_org_new_p['MedR']+1/metrics_rank_gen_new_p['MedR'])), 2)
delta_MeanR_new_p = round(200*((1/metrics_rank_org_new_p['MeanR']-1/metrics_rank_gen_new_p['MeanR'])/(1/metrics_rank_org_new_p['MeanR']+1/metrics_rank_gen_new_p['MeanR'])), 2)
csv_data_p.append(["delta",delta_R1_p,delta_R5_p,delta_R10_p,delta_MedR_p,delta_MeanR_p,round(((delta_R1_p+delta_MeanR_p)/2), 2),round(((delta_R1_p+delta_MedR_p+delta_MeanR_p)/3), 2)])
csv_data_p.append(["delta_new",delta_R1_new_p,delta_R5_new_p,delta_R10_new_p,delta_MedR_new_p,delta_MeanR_new_p,round((delta_R1_new_p+delta_MeanR_new_p)/2,2),round((delta_R1_new_p+delta_MedR_new_p+delta_MeanR_new_p)/3, 2)])
csv_data_p.append(["ΔΔ",round(-delta_R1_new_p+delta_R1_p, 2),
round(-delta_R5_new_p+delta_R5_p, 2),
round(-delta_R10_new_p+delta_R10_p, 2),
round(-delta_MedR_new_p+delta_MedR_p, 2),
round(-delta_MeanR_new_p+delta_MeanR_p, 2),
round(-((delta_R1_new_p+delta_MeanR_new_p)/2+(-delta_R1_p-delta_MeanR_p)/2), 2),
round(-((delta_R1_new_p+delta_MedR_new_p+delta_MeanR_new_p)/3+(-delta_R1_p-delta_MedR_p-delta_MeanR_p))/3, 2)])

csv_filename = '/data/gaohaowen/workspace/InternVideo-main/InternVideo1/vector_debias_sim/org.csv'
csv_filename_p = '/data/gaohaowen/workspace/InternVideo-main/InternVideo1/vector_debias_sim/p_debias.csv'

with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(csv_data)

with open(csv_filename_p, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(csv_data_p)