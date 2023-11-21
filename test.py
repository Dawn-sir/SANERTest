import os
import csv

import pandas as pd


def emb_to_csv():
    # CPG 文件夹路径
    cpg_folder = './cpg'

    # 遍历子文件夹
    for i in range(4467):
        subfolder_path = os.path.join(cpg_folder, str(i))

        # 检查子文件夹是否存在
        if not os.path.exists(subfolder_path):
            continue

        # 查找 .emb 文件
        emb_files = [f for f in os.listdir(subfolder_path) if f.endswith('.emb')]

        # 遍历 .emb 文件
        for emb_file in emb_files:
            emb_file_path = os.path.join(subfolder_path, emb_file)

            # 读取 .emb 文件并转换为 .csv 文件
            with open(emb_file_path, 'r') as emb_file:
                csv_file_path = emb_file_path.replace('.emb', '.csv')
                csv_writer = csv.writer(open(csv_file_path, 'w', newline=''))

                # 逐行读取 .emb 文件并写入 .csv 文件, 但不包含第一行和第一列
                falg = True
                for line in emb_file:
                    if falg:
                        falg = False
                        continue
                    csv_writer.writerow(line.strip().split()[1:])

            print(f"转换文件 {emb_file_path} 为 {csv_file_path}")


def merge_csv_files(folder_path, output_file):
    # 创建一个空的列表来存储数据
    data = []

    # 遍历子文件夹编号范围
    for i in range(4467):
        # 构建子文件夹路径
        subfolder_path = os.path.join(folder_path, str(i))

        # 检查子文件夹是否存在
        if os.path.exists(subfolder_path):
            # 遍历子文件夹中的文件
            for file in os.listdir(subfolder_path):
                # 检查文件扩展名是否为 .csv
                if file.endswith('.csv'):
                    # 构建文件路径
                    file_path = os.path.join(subfolder_path, file)

                    # 使用适当的方法读取 .csv 文件的内容
                    with open(file_path, 'r') as csvfile:
                        reader = csv.reader(csvfile)
                        rows = list(reader)

                    # 在每一行的第一个位置添加子文件夹编号
                    for row in rows:
                        row.insert(0, str(i))

                    # 将数据添加到列表中
                    data.extend(rows)

    # 将数据保存到输出文件中
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # 写入数据
        writer.writerows(data)

    print(f"已将数据保存到 {output_file}")


# 写一个函数，实现合并csv文件中第二列以后的所有列为一个列表，保存在原来的第二列
def merge_columns(input_file, output_file):
    # 读取输入文件的内容
    with open(input_file, 'r') as csvfile:
        reader = csv.reader(csvfile)

        # 遍历每一行，将第二列以后的所有列元素转换为float类型后，合并成一个向量，保存在第二列
        # 创建一个dataframe，第一列列名为id，第二列列名为vector
        data = {"id": [], "vector": []}
        df = pd.DataFrame(data)

        # 遍历每一行
        for row in reader:
            if row[1] == 'x_0':
                continue
            # 将第一列添加到新的行中
            new_row = []
            # 遍历第二列以后的所有列
            for item in row[1:]:
                # 将字符串转换为浮点数，并添加到新的行中
                item = item.replace('[', '').replace(']', '')
                tmp = 0.0
                for i in item.split(', '):
                    tmp += float(i.strip('\''))
                new_row.append(tmp)
                # 将新的行添加到数据列表中
            df = df.append({"id": row[0], "vector": new_row}, ignore_index=True)

    # 将df保存到csv文件中
    df.to_csv(output_file, index=False)

    print(f"已将数据保存到 {output_file}")


# 写一个函数，读取csv文件，并打印前面的5行
def read_csv_file(file_path):
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = list(reader)
        for row in rows[:5]:
            print(row)

# 写一个函数，比较graph.csv文件中filename一列和cpg.csv文件中id列相同的行，将graph.csv文件中的第2到第8列的数据保存到cpg.csv文件中相应的行中
def compare_and_save_data(graph_file, cpg_file, output_file):
    # 读取 graph.csv 文件
    graph_df = pd.read_csv(graph_file)
    # 读取 cpg.csv 文件
    cpg_df = pd.read_csv(cpg_file)
    # 将 graph.csv 文件中 filename 列和 cpg.csv 文件中 id 列相同的行进行匹配
    merged_df = pd.merge(graph_df, cpg_df, left_on='filename', right_on='id')
    # 删掉merged_df中的filename列和codebert列
    merged_df.drop(['id', 'codebert'], axis=1, inplace=True)
    merged_df.to_csv(output_file, index=False)


def IndexReflection1(valid_file, test_file, output_file):
    idx_vlaid = []
    # valid_file每一行一个数字，将读取到的数字存储到idx列表中
    with open(valid_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            idx_vlaid.append(int(row[0]))

    idx_test = []
    # test_file每一行一个数字，将读取到的数字存储到idx_test列表中
    with open(test_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            idx_test.append(int(row[0]))

    res = []
    for i in range(len(idx_vlaid)):
        res.append(idx_test[idx_vlaid[i]])
    print(len(idx_vlaid))
    print(res)
    print(len(res))
    # 将res保存到output_file中
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for i in range(len(res)):
            writer.writerow([res[i]])


def CheckSize(test_file, valid_file, train_file):
    idx_vlaid = []
    # valid_file每一行一个数字，将读取到的数字存储到idx列表中
    with open(valid_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            idx_vlaid.append(int(row[0]))

    idx_test = []
    # test_file每一行一个数字，将读取到的数字存储到idx_test列表中
    with open(test_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            idx_test.append(int(row[0]))

    idx_train = []
    # train_file每一行一个数字，将读取到的数字存储到idx_train列表中
    with open(train_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            idx_train.append(int(row[0]))

    res = []
    # 合并idx_train、idx_vlaid、idx_test
    res.extend(idx_train)
    res.extend(idx_vlaid)
    res.extend(idx_test)
    print(len(res))
    print(len(set(res)))

def IndexReflection2(label_file, valid_file, test_file, output_file):
    idx_label = []
    # label_file每一行一个数字，将读取到的数字存储到idx_label列表中
    with open(label_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            idx_label.append(int(row[0]))

    idx_vlaid = []
    # valid_file每一行一个数字，将读取到的数字存储到idx_vlaid列表中
    with open(valid_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            idx_vlaid.append(int(row[0]))

    idx_test = []
    # test_file每一行一个数字，将读取到的数字存储到idx_test列表中
    with open(test_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            idx_test.append(int(row[0]))

    res = []
    for i in range(len(idx_label)):
        res.append(idx_test[idx_vlaid[idx_label[i]]])
    print(len(idx_label))
    print(res)
    print(len(res))
    # 将res保存到output_file中
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for i in range(len(res)):
            writer.writerow([res[i]])


def CheckSize2(label_file, unlabel_file, valid_file, test_file):
    idx_label = []
    # label_file每一行一个数字，将读取到的数字存储到idx_label列表中
    with open(label_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            idx_label.append(int(row[0]))

    idx_unlabel = []
    # unlabel_file每一行一个数字，将读取到的数字存储到idx_unlabel列表中
    with open(unlabel_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            idx_unlabel.append(int(row[0]))

    idx_vlaid = []
    # valid_file每一行一个数字，将读取到的数字存储到idx_vlaid列表中
    with open(valid_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            idx_vlaid.append(int(row[0]))

    idx_test = []
    # test_file每一行一个数字，将读取到的数字存储到idx_test列表中
    with open(test_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            idx_test.append(int(row[0]))

    res = []
    # 合并idx_label、idx_unlabel、idx_vlaid、idx_test
    res.extend(idx_label)
    res.extend(idx_unlabel)
    res.extend(idx_vlaid)
    res.extend(idx_test)
    print(len(res))
    print(len(set(res)))

def transform_embedding_colums(file_path):
    data = pd.read_csv(file_path)
    # 将'embeeding'列转换成以逗号分隔的字符串，并删去回车符
    data['embedding'] = data['embedding'].apply(lambda x: x.replace('\n', ' '))
    # 将'embeeding'列连续的空格转换成单个空格
    data['embedding'] = data['embedding'].apply(lambda x: ' '.join(x.split()))
    # 去除'embeeding'列中的[后面可能存在的空格
    data['embedding'] = data['embedding'].apply(lambda x: x.replace('[ ', '['))
    data['embedding'] = data['embedding'].apply(lambda x: x.replace('] ', ']'))

    # 将'embeeding'列空格转换成逗号
    data['embedding'] = data['embedding'].apply(lambda x: x.replace(' ', ', '))

    # 将修改后的数据保存到csv文件中的'embeeding'列
    data.to_csv(file_path, index=False)


def node2vec(filepath):
    # 读取文件
    data = pd.read_csv(filepath)
    # 去除Embedding列中第一个字符'['以及'array('以及'dtype=float32)'以及最后一个字符']'
    data['Embedding'] = data['Embedding'].apply(lambda x: x[1:-1])
    data['Embedding'] = data['Embedding'].apply(lambda x: x.replace('array(', ''))
    data['Embedding'] = data['Embedding'].apply(lambda x: x.replace('dtype=float32)', ''))
    data['Embedding'] = data['Embedding'].apply(lambda x: x[:-1])
    # 去除最后一个字符','
    data['Embedding'] = data['Embedding'].apply(lambda x: x[:-7])
    # 去除Embedding列中的回车符
    data['Embedding'] = data['Embedding'].apply(lambda x: x.replace('\n', ''))
    data['Embedding'] = data['Embedding'].apply(lambda x: ' '.join(x.split()))

    # 打印第一行的Embedding列
    print(data['Embedding'][0])

    # 保存到csv文件中
    data.to_csv(filepath, index=False)


def add_cvss(filepath):
    # 读取文件
    data = pd.read_csv(filepath)
    data_add = pd.read_csv('./graph_code_feature_pool_3.csv')
    # 取出data_add的第二列至第八列全部数据
    data_add = data_add.iloc[:, 1:8]
    # print(data_add)
    # 将data_add的数据添加到data中
    data = pd.concat([data, data_add], axis=1)
    print(data)
    # 保存到csv文件中
    data.to_csv(filepath, index=False)

    
if __name__ == '__main__':
    # emb_to_csv()
    # merge_csv_files('./cpg', './cpg.csv')
    # merge_columns('./cpg.csv', './cpg.csv')
    # compare_and_save_data('./graphcodebert_feature_pool.csv', './cpg.csv', './cpg_new.csv')
    # read_csv_file('./cpg_new.csv')

    # IndexReflection1(valid_file='valid_ind_1.txt', test_file='test_ind_2.txt', output_file='new_valid_ind_1.txt')
    # IndexReflection1(valid_file='valid_ind_2.txt', test_file='test_ind_2.txt', output_file='new_valid_ind_2.txt')
    # CheckSize(test_file='test_ind_1.txt', valid_file='new_valid_ind_1.txt', train_file='new_valid_ind_2.txt')
    # IndexReflection2(label_file='labeled_ind_1.txt', valid_file='valid_ind_2.txt', test_file='test_ind_2.txt', output_file='new_labeled_ind_1.txt')
    # IndexReflection2(label_file='labeled_ind_2.txt', valid_file='valid_ind_2.txt', test_file='test_ind_2.txt', output_file='new_labeled_ind_2.txt')
    # CheckSize2(test_file='test_ind_1.txt', valid_file='new_valid_ind_1.txt', label_file='new_labeled_ind_1.txt', unlabel_file='new_labeled_ind_2.txt')

    # merge_columns('./g2v_feature.csv', './new_g2v_feature.csv')
    # merge_columns('./g2v_feature_256.csv', './new_g2v_feature_256.csv')

    transform_embedding_colums('./gcndata2.csv')

    # node2vec('./node2vec.csv')
    # add_cvss('./node2vec.csv')



