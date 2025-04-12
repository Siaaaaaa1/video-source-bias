import numpy as np
import scipy.stats
import logging
import os
import pandas as pd
import csv

# 日志文件路径
log_file_path = '/data/gaohaowen/workspace/logging.log'

# 创建logger
logger = logging.getLogger('logger')
logger.setLevel(logging.INFO)  # 设置日志记录的最低级别

# 创建用于写入日志文件的FileHandler
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.INFO)  # 设置FileHandler的日志级别

# 创建日志格式
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)

# 将FileHandler添加到logger
logger.addHandler(file_handler)

# 搜索所有符合要求的文件夹名称，返回文件夹列表
def find_and_rename_folders(root_dir):
    matching_folders = []
    # 遍历根目录下的所有文件夹
    for subdir, dirs, files in os.walk(root_dir):
        for folder in dirs:
            folder_path = os.path.join(subdir, folder)
            # 检查文件夹中的文件
            files_in_folder = os.listdir(folder_path)
            # 检查文件组合
            if ('sim_dsl_org.npy' in files_in_folder) and ('sim_dsl_gen.npy' in files_in_folder):
                # 重命名文件夹
                new_folder_path_gen = os.path.join(folder_path, 'sim_gen.npy')
                new_folder_path_org = os.path.join(folder_path, 'sim_org.npy')
                old_folder_path_gen = os.path.join(folder_path, 'sim_dsl_gen.npy')
                old_folder_path_org = os.path.join(folder_path, 'sim_dsl_org.npy')
                os.rename(old_folder_path_gen, new_folder_path_gen)
                os.rename(old_folder_path_org, new_folder_path_org)
                matching_folders.append(folder_path)
            elif ('sim_org.npy' in files_in_folder) and ('sim_gen.npy' in files_in_folder):
                matching_folders.append(folder_path)
    return sorted(matching_folders)

def load_numpy_array(filename):
    """
    从文件中加载NumPy数组。
    
    参数:
    filename -- 包含NumPy数组的文件名。
    
    返回:
    加载的NumPy数组。
    """
    array = np.load(filename,allow_pickle=True)
    print(f"从文件 {filename} 加载数组成功")
    return array

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

def ranks2metrics(cols, num_queries):
    """
    计算给定列向量中元素的排名指标，包括传统的Ranking指标和基于倒数及对数倒数的指标。

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
    - R1/1/x: 基于倒数的指标，对于每个位置x，计算1/(x+1)的总和，然后除以查询总数。
    - R3/1/x: 针对前3个位置。
    - R5/1/x: 针对前5个位置。
    - R10/1/x: 针对前10个位置。
    - R1/1/log2x: 基于对数倒数的指标，对于每个位置x，计算1/(log2(x+1))的总和，然后除以查询总数。
    - R3/1/log2x: 针对前3个位置。
    - R5/1/log2x: 针对前5个位置。
    - R10/1/log2x: 针对前10个位置。

    功能:
    该函数计算并返回一个字典，其中包含查询排名的不同指标。
    这些指标通常用于评估搜索系统或推荐系统的性能，特别是考虑到位置权重的倒数和对数倒数。
    """
    metrics = {}
    metrics["R1"] = 100 * float(np.sum(cols == 0)) / num_queries
    metrics["R3"] = 100 * float(np.sum(cols < 3)) / num_queries
    metrics["R5"] = 100 * float(np.sum(cols < 5)) / num_queries
    metrics["R10"] = 100 * float(np.sum(cols < 10)) / num_queries
    metrics["R50"] = 100 * float(np.sum(cols < 50)) / num_queries
    metrics["MedR"] = np.median(cols) + 1
    metrics["MeanR"] = np.mean(cols) + 1
    metrics["R1/1/x"] = invert_numbers(cols, 1)/num_queries
    metrics["R3/1/x"] = invert_numbers(cols, 3)/num_queries
    metrics["R5/1/x"] = invert_numbers(cols, 5)/num_queries
    metrics["R10/1/x"] = invert_numbers(cols, 10)/num_queries
    metrics["R1/1/log2x"] = compute_log2_inverse(cols, 1)/num_queries
    metrics["R3/1/log2x"] = compute_log2_inverse(cols, 3)/num_queries
    metrics["R5/1/log2x"] = compute_log2_inverse(cols, 5)/num_queries
    metrics["R10/1/log2x"] = compute_log2_inverse(cols, 10)/num_queries
    return metrics
    

# 利用rank矩阵计算指标：1/x
def invert_numbers(arr,group_size):
    # 创建一个新数组来存储倒数
    arr = arr + 1
    if group_size == 0:
        raise ValueError("group_size cannot be zero")
    # 将数组中的元素除以group_size，然后向上取整
    result = np.ceil(arr / group_size).astype(int)  # 转换结果为整数
    inverted_arr = []
    for num in result:
        if num != 0:
            inverted_arr.append(1 / num)
        else:
            # 如果数组中的数为0，可以将其设为None或者其他标记值
            inverted_arr.append(0)
    return sum(inverted_arr)

# 利用rank矩阵计算指标：1/(log2(x)+1)
def compute_log2_inverse(arr,group_size):
    # 创建一个新数组来存储倒数
    arr = arr + 1
    if group_size == 0:
        raise ValueError("group_size cannot be zero")
    # 将数组中的元素除以group_size，然后向上取整
    arr = np.ceil(arr / group_size).astype(int)  # 转换结果为整数
    # 避免对0和负数取对数
    result = np.where(arr > 0, 1 / (np.log2(arr + 1)), 0)
    return result.sum()

# 将CSV文件转换为Excel文件
def csv_to_excel(csv_filename, excel_filename):
    # 读取CSV文件
    df = pd.read_csv(csv_filename)
    
    # 将DataFrame保存为Excel文件
    df.to_excel(excel_filename, index=False)
    print(f"Excel file '{excel_filename}' has been created.")


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

import argparse
fold_path = "/data/gaohaowen/workspace/ZGET_METRIC"
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='创建logging')
    parser.add_argument('-f', '--fold', help='文件夹名称')
    args = parser.parse_args()
    fold_path = os.path.join(fold_path, args.fold)
    np.set_printoptions(threshold=np.inf)
    sub_folds = find_and_rename_folders(fold_path)
    for fold in sub_folds:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()
        log_file_path = os.path.join(fold,'logging.log')
        # os.remove(os.path.join(fold,'logging.log'))
        sim_gen = load_numpy_array(os.path.join(fold,'sim_gen.npy'))
        sim_org = load_numpy_array(os.path.join(fold,'sim_org.npy'))
        logger = logging.getLogger('logger')
        logger.setLevel(logging.INFO)  # 设置日志记录的最低级别
        # 创建用于写入日志文件的FileHandler
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.INFO)  # 设置FileHandler的日志级别
        # 创建日志格式
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        # 将FileHandler添加到logger
        logger.addHandler(file_handler)
        rank = rank_in_rows(np.hstack((sim_org,sim_gen)))
        rank_org = rank_in_rows(sim_org)
        rank_gen = rank_in_rows(sim_gen)
        sinlge_size = rank.shape[0]
        metrics_org = cols2metrics(np.diag(rank[:,:sinlge_size]),sinlge_size)
        metrics_gen = cols2metrics(np.diag(rank[:,sinlge_size:]),sinlge_size)
        metrics_rank_org = ranks2metrics(np.diag(rank_org),sinlge_size)
        metrics_rank_gen = ranks2metrics(np.diag(rank_gen),sinlge_size)
        # position rank
        rank_position_gen1, rank_position_org1 = get_rank_new(np.diag(rank_gen),np.diag(rank_org),0)
        rank_position_gen2, rank_position_org2 = get_rank_new(np.diag(rank_gen),np.diag(rank_org),1)        
        rank_position_gen1 = rank_position_gen1-1
        rank_position_org1 = rank_position_org1-1
        rank_position_gen2 = rank_position_gen2-1
        rank_position_org2 = rank_position_org2-1
        metrics_position_gen1 = cols2metrics(rank_position_gen1,sinlge_size)
        metrics_position_org1 = cols2metrics(rank_position_org1,sinlge_size)
        metrics_position_gen2 = cols2metrics(rank_position_gen2,sinlge_size)
        metrics_position_org2 = cols2metrics(rank_position_org2,sinlge_size)
        metrics_rank_gen_new = average_values(metrics_position_gen1, metrics_position_gen2)
        metrics_rank_org_new = average_values(metrics_position_org1, metrics_position_org2)

        
        csv_data = []
        path = fold
        print(fold)
        parent, folder = os.path.split(path)
        csv_data.append([folder])
        msg_single_org = (f"\nR1: {metrics_rank_org['R1']:.1f}, "
                    f"R5: {metrics_rank_org['R5']:.1f}, "
                    f"R10: {metrics_rank_org['R10']:.1f}, "
                    f"MedR: {metrics_rank_org['MedR']}, "
                    f"MeanR: {metrics_rank_org['MeanR']}")
        csv_data.append(["org_single",metrics_rank_org['R1'],metrics_rank_org['R5'],metrics_rank_org['R10'],metrics_rank_org['MedR'],metrics_rank_org['MeanR'],0,0])
        msg_single_gen = (f"\nR1: {metrics_rank_gen['R1']:.1f}, "
                    f"R5: {metrics_rank_gen['R5']:.1f}, "
                    f"R10: {metrics_rank_gen['R10']:.1f}, "
                    f"MedR: {metrics_rank_gen['MedR']}, "
                    f"MeanR: {metrics_rank_gen['MeanR']}")
        csv_data.append(["gen_single",metrics_rank_gen['R1'],metrics_rank_gen['R5'],metrics_rank_gen['R10'],metrics_rank_gen['MedR'],metrics_rank_gen['MeanR'],0,0])
        msg_org = (f"\nR1: {metrics_org['R1']:.1f}, "
                    f"R5: {metrics_org['R5']:.1f}, "
                    f"R10: {metrics_org['R10']:.1f}, "
                    f"MedR: {metrics_org['MedR']}, "
                    f"MeanR: {metrics_org['MeanR']}")
        csv_data.append(["org",metrics_org['R1'],metrics_org['R5'],metrics_org['R10'],metrics_org['MedR'],metrics_org['MeanR'],0,0])
        msg_gen = (f"\nR1: {metrics_gen['R1']:.1f}, "
                    f"R5: {metrics_gen['R5']:.1f}, "
                    f"R10: {metrics_gen['R10']:.1f}, "
                    f"MedR: {metrics_gen['MedR']}, "
                    f"MeanR: {metrics_gen['MeanR']}")
        csv_data.append(["gen",metrics_gen['R1'],metrics_gen['R5'],metrics_gen['R10'],metrics_gen['MedR'],metrics_gen['MeanR'],0,0])
        msg_org_new = (f"\nR1: {metrics_rank_org_new['R1']:.1f}, "
                    f"R5: {metrics_rank_org_new['R5']:.1f}, "
                    f"R10: {metrics_rank_org_new['R10']:.1f}, "
                    f"MedR: {metrics_rank_org_new['MedR']}, "
                    f"MeanR: {metrics_rank_org_new['MeanR']}")
        csv_data.append(["org_new",metrics_rank_org_new['R1'],metrics_rank_org_new['R5'],metrics_rank_org_new['R10'],metrics_rank_org_new['MedR'],metrics_rank_org_new['MeanR'],0,0])
        msg_gen_new = (f"\nR1: {metrics_rank_gen_new['R1']:.1f}, "
                    f"R5: {metrics_rank_gen_new['R5']:.1f}, "
                    f"R10: {metrics_rank_gen_new['R10']:.1f}, "
                    f"MedR: {metrics_rank_gen_new['MedR']}, "
                    f"MeanR: {metrics_rank_gen_new['MeanR']}")
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
        delta_msg = (f"\nR1: {delta_R1:.2f}, "
                    f"R3: {delta_R3:.2f}, "
                    f"R5: {delta_R5:.2f}, "
                    f"R10: {delta_R10:.2f}, "
                    f"MedR: {delta_MedR:.2f}, "
                    f"MeanR: {delta_MeanR:.2f}, "
                    f"R1+MeanR: {((delta_R1+delta_MeanR)/2):.2f}, "
                    f"R1+MedR+MeanR: {((delta_R1+delta_MedR+delta_MeanR)/3):.2f}")
        csv_data.append(["delta",delta_R1,delta_R5,delta_R10,delta_MedR,delta_MeanR,round(((delta_R1+delta_MeanR)/2), 2),round(((delta_R1+delta_MedR+delta_MeanR)/3), 2)])
        # csv_data.append(["delta",delta_R1,delta_MedR,delta_MeanR,round(((delta_R1+delta_MeanR)/2), 2),round(((delta_R1+delta_MedR+delta_MeanR)/3), 2)])
        delta_msg_new = (f"\nR1: {delta_R1_new:.2f}, "
                    f"R3: {delta_R3_new:.2f}, "
                    f"R5: {delta_R5_new:.2f}, "
                    f"R10: {delta_R10_new:.2f}, "
                    f"MedR: {delta_MedR_new:.2f}, "
                    f"MeanR: {delta_MeanR_new:.2f}, "
                    f"R1+MeanR: {((delta_R1_new+delta_MeanR_new)/2):.2f}, "
                    f"R1+MedR+MeanR: {((delta_R1_new+delta_MedR_new+delta_MeanR_new)/3):.2f}")
        csv_data.append(["delta_new",delta_R1_new,delta_R5_new,delta_R10_new,delta_MedR_new,delta_MeanR_new,round((delta_R1_new+delta_MeanR_new)/2,2),round((delta_R1_new+delta_MedR_new+delta_MeanR_new)/3, 2)])
        # csv_data.append(["delta_new",delta_R1_new,delta_MedR_new,delta_MeanR_new,round((delta_R1_new+delta_MeanR_new)/2,2),round((delta_R1_new+delta_MedR_new+delta_MeanR_new)/3, 2)])
        delta_delta_msg = (f"\nR1: {-delta_R1_new+delta_R1:.2f}, "
                    f"R5: {-delta_R5_new+delta_R5:.2f}, "
                    f"R10: {-delta_R10_new+delta_R10:.2f}, "
                    f"MedR: {-delta_MedR_new+delta_MedR:.2f}, "
                    f"MeanR: {-delta_MeanR_new+delta_MeanR:.2f}, "
                    f"R1+MeanR: {-((delta_R1_new+delta_MeanR_new)/2+(-delta_R1-delta_MeanR)/2):.2f}, "
                    f"R1+MedR+MeanR: {-((delta_R1_new+delta_MedR_new+delta_MeanR_new)/3+(-delta_R1-delta_MedR-delta_MeanR)/3):.2f}")
        csv_data.append(["ΔΔ",round(-delta_R1_new+delta_R1, 2),
                         round(-delta_R5_new+delta_R5, 2),
                         round(-delta_R10_new+delta_R10, 2),
                         round(-delta_MedR_new+delta_MedR, 2),
                         round(-delta_MeanR_new+delta_MeanR, 2),
                         round(-((delta_R1_new+delta_MeanR_new)/2+(-delta_R1-delta_MeanR)/2), 2),
                         round(-((delta_R1_new+delta_MedR_new+delta_MeanR_new)/3+(-delta_R1-delta_MedR-delta_MeanR)/3), 2)])
        # csv_data.append(["ΔΔ",round(-delta_R1_new+delta_R1, 2),
        #                  round(-delta_MedR_new+delta_MedR, 2),
        #                  round(-delta_MeanR_new+delta_MeanR, 2),
        #                  round(-((delta_R1_new+delta_MeanR_new)/2+(-delta_R1-delta_MeanR)/2), 2),
        #                  round(-((delta_R1_new+delta_MedR_new+delta_MeanR_new)/3+(-delta_R1-delta_MedR-delta_MeanR))/3, 2)])
        logger.info("org_single:"+str(msg_single_org))
        logger.info("gen_single:"+str(msg_single_gen))
        logger.info("org:"+str(msg_org))
        logger.info("gen:"+str(msg_gen))
        logger.info("org_new:"+str(msg_org_new))
        logger.info("gen_new:"+str(msg_gen_new))
        logger.info("delta:"+str(delta_msg))
        logger.info("delta_new:"+str(delta_msg_new))
        logger.info("delta_delta:"+str(delta_delta_msg))
        # logger.info("org/rank:"+str(rank_msg_org))
        # logger.info("gen/rank:"+str(rank_msg_gen))
        # logger.info("div/rank:"+str(rank_msg_div))
        # logger.info("div/MedR:"+str(div_MedR))
        # logger.info("div/MeanR:"+str(div_MeanR))
        # logger.info("merge/rank:"+str(rank_msg_merge))
        csv_filename = '/data/gaohaowen/workspace/'+args.fold+'.csv'
        if os.path.isfile(csv_filename):
            with open(csv_filename, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(csv_data)
        else:
            with open(csv_filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(csv_data)
        # 读取CSV文件
        df = pd.read_csv("/data/gaohaowen/workspace/"+args.fold+".csv")

        # 将DataFrame写入Excel文件
        df.to_excel("/data/gaohaowen/workspace/"+args.fold+".xlsx")