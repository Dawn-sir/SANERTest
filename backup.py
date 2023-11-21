df_method = pd.read_parquet('E:\桌面\LAMDA-SSL-master\method_whole_codebert.parquet')

print('Loaded data')
selected_cols = ['codebert', 'cvss2_confidentiality_impact']
df_tmp = df_method[selected_cols].copy()
print('Copy success!')

# 列值映射
column_name = 'cvss2_confidentiality_impact'
# 替换为实际的列名
mapping = {'NONE': 0, 'PARTIAL': 1, 'COMPLETE': 2}
# 设置映射关系
# 使用map函数进行映射
df_tmp[column_name] = df_tmp[column_name].map(mapping)
# df_tmp['codebert'] = df_tmp['codebert'].str.decode('utf-8').str.strip("b''")

# 随机选取训练集
train_data = df_tmp.sample(frac=0.8, random_state=1)
print('train_data',train_data.shape)
# 从原始数据集中删除训练集数据
df_tmp = df_tmp.drop(train_data.index)

# 随机选取测试集
test_data = df_tmp.sample(frac=0.5, random_state=2)
print('test_data',test_data.shape)
# 从原始数据集中删除测试集数据
df_tmp = df_tmp.drop(test_data.index)

# 剩余数据作为验证集
valid_data = df_tmp
print('valid_data', valid_data.shape)

from LAMDA_SSL.Split.DataSplit import DataSplit

train_X = train_data['codebert'].values
# print(len(X[0].decode('utf-8')[1:-1].split(", "))) 打印宽度
##### 训练集：将bytes类型转化为ndarray类型
X_array = np.empty((0,768))
for i, row in enumerate(train_X):
    s = row.decode('utf-8')
    s = s[1:-1]
    # print(s)
    num_list = s.split(", ")
    # print(len(num_list)) !!!
    num_list = [float(num) for num in num_list]
    # print(num_list)
    num_array = np.array(num_list)
    # print(num_array.shape)
    num_array = num_array.reshape(1, -1)
    # print(num_array.shape)
    X_array = np.append(X_array, num_array, axis=0)

y = train_data['cvss2_confidentiality_impact'].values
print(X_array.shape)
print(y.shape)
##### 分隔数据集
labeled_X, labeled_y, unlabeled_X, unlabeled_y = DataSplit(X=X_array, y=y, size_split=0.8,
                                                           stratified=True, shuffle=True,
                                                           random_state=0)

##### 测试集：将bytes类型转化为ndarray类型
test_X = test_data['codebert'].values
print('test_X',test_X.shape)
print(len(test_X[0].decode('utf-8')[1:-1].split(", ")))
test_y = test_data['cvss2_confidentiality_impact'].values
test_X_array = np.empty((0,768))
for i, row in enumerate(test_X):
    s = row.decode('utf-8')
    s = s[1:-1]
    # print(s)
    num_list = s.split(", ")
    # print(len(num_list)) !!!
    num_list = [float(num) for num in num_list]
    # print(num_list)
    num_array = np.array(num_list)
    # print(num_array.shape)
    num_array = num_array.reshape(1, -1)
    # print(num_array.shape)
    test_X_array = np.append(test_X_array, num_array, axis=0)


print('成功划分数据集')
# print(type(train_data['codebert']))
# 将训练集、测试集和验证集转换为NumPy数组或Pandas DataFrame
# train_X, train_y = np.array(train_data['codebert']), np.array(train_data['cvss2_confidentiality_impact'])
# test_X, test_y = np.array(test_data['codebert']), np.array(test_data['cvss2_confidentiality_impact'])
# valid_X, valid_y = np.array(valid_data['codebert']), np.array(valid_data['cvss2_confidentiality_impact'])
#
# print('成功转化成numpy数组')
# print(train_X)
# print(train_y)
#
# print(type(train_y))
# print(train_X.shape)
# print(train_y.shape)