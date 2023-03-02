import numpy as np
import math

"""
RNA相似度
"""
def GetDiseaseId(inMat, miRNAId, disease_length):
    DiseaseId = []
    for i in range(disease_length):
        if inMat.T[miRNAId][i] == 1:
            DiseaseId.append(i)
    return DiseaseId
"""
Jaccard距离
"""
def jaccard_similarity_miRNA(intMat, miRNA_length, disease_length):
        similarity = []
        for i in range(miRNA_length):
            # print("第", i, "回合")
            temp = []
            i_count = len(GetDiseaseId(intMat, i, disease_length))
            for j in range(miRNA_length):
                j_count = len(GetDiseaseId(intMat, j, disease_length))
                intersection = len(list(set(GetDiseaseId(intMat, i, disease_length)) & set(GetDiseaseId(intMat, j, disease_length))))  # 交集
                i_j_union_set = i_count + j_count - intersection
                if i_j_union_set == 0:
                    Calculated_similarity = 0
                else:
                    Calculated_similarity = intersection/i_j_union_set
                temp.append(format(int(Calculated_similarity*10000)/10000, '.4f'))
                #  format(1.23456, '.2f')
            similarity.append(temp)
            Similarity = np.array(similarity, dtype=np.float64)
        return Similarity

""""
基于物品
"""
def itemcf_similarity(intMat, miRNA_length, disease_length):
    similarity = []
    for i in range(miRNA_length):
        # print("第", i, "回合")
        temp = []
        i_count = len(GetDiseaseId(intMat, i, disease_length))
        for j in range(miRNA_length):
            j_count = len(GetDiseaseId(intMat, j, disease_length))
            intersection = len(list(set(GetDiseaseId(intMat, i, disease_length)) & set(GetDiseaseId(intMat, j, disease_length))))  # 交集
            if i_count == 0 or j_count == 0:
                Calculated_similarity = 0
            else:
                Calculated_similarity = intersection / math.sqrt(i_count * j_count)
            temp.append(format(int(Calculated_similarity * 10000) / 10000, '.4f'))
            #  format(1.23456, '.2f')
        similarity.append(temp)
        Similarity = np.array(similarity, dtype=np.float64)
    return Similarity

"""
基于疾病
"""
def GetMicRNAId(inMat, DiseaseId, miRNA_length):
    micRNAId = []
    for i in range(miRNA_length):
        if inMat.T[i][DiseaseId] == 1:
            micRNAId.append(i)
    return micRNAId

# 最终输出一个len(disease)*len(disease)的相似矩阵
#  383*383
def usercf_Similarity(intMat, miRNA_length, disease_length):
    similarity = []
    for i in range(disease_length):
        # print("第", i, "回合")
        temp = []
        i_count = len(GetMicRNAId(intMat, i, miRNA_length))
        for j in range(disease_length):
            j_count = len(GetMicRNAId(intMat, j, miRNA_length))
            intersection = len(list(set(GetMicRNAId(intMat, i, miRNA_length)) & set(GetMicRNAId(intMat, j, miRNA_length))))  # 交集
            if i_count == 0 or j_count == 0:
                Calculated_similarity = 0
            else:
                Calculated_similarity = intersection / math.sqrt(i_count * j_count)
            temp.append(format(int(Calculated_similarity * 10000) / 10000, '.4f'))
        similarity.append(temp)
        Similarity = np.array(similarity, dtype=np.float64)
    return Similarity

def jaccard_similarity_disease(intMat, miRNA_length, disease_length):
    similarity = []
    for i in range(disease_length):
        # print("第", i, "回合")
        temp = []
        i_count = len(GetMicRNAId(intMat, i, miRNA_length))
        for j in range(disease_length):
            j_count = len(GetMicRNAId(intMat, j, miRNA_length))
            intersection = len(list(
                set(GetMicRNAId(intMat, i, miRNA_length)) & set(GetMicRNAId(intMat, j, miRNA_length))))  # 交集
            i_j_union_set = i_count + j_count - intersection
            if i_j_union_set == 0:
                Calculated_similarity = 0
            else:
                Calculated_similarity = intersection / i_j_union_set
            temp.append(format(int(Calculated_similarity * 10000) / 10000, '.4f'))
            #  format(1.23456, '.2f')
        similarity.append(temp)
        Similarity = np.array(similarity, dtype=np.float64)
    return Similarity

