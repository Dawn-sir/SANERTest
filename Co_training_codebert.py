from LAMDA_SSL.Dataset.SemiDataset import SemiDataset
from LAMDA_SSL.Base.TabularMixin import TabularMixin
from LAMDA_SSL.Split.DataSplit import DataSplit
from LAMDA_SSL.Dataset.TrainDataset import TrainDataset
from LAMDA_SSL.Dataset.LabeledDataset import LabeledDataset
from LAMDA_SSL.Dataset.UnlabeledDataset import UnlabeledDataset
from sklearn import datasets
import numpy as np
import pandas as pd
import ast

class Co_training_codebert(SemiDataset,TabularMixin):
    def __init__(
        self,
        default_transforms=False,
        pre_transform=None,
        transforms=None,
        transform = None,
        target_transform = None,
        unlabeled_transform=None,
        valid_transform=None,
        test_transform=None,
        test_size=None,
        valid_size=None,
        labeled_size=0.1,
        stratified=False,
        shuffle=True,
        random_state=None,
        class_name=None,
        column_name=None
    ) -> None:
        self.column_name = None
        self.class_name = None
        self.default_transforms=default_transforms
        self.labeled_X=None
        self.labeled_y=None
        self.unlabeled_X=None
        self.unlabeled_y=None
        self.valid_X=None
        self.valid_y=None
        self.test_X=None
        self.test_y=None

        self.labeled_dataset=None
        self.unlabeled_dataset=None
        self.train_dataset=None
        self.valid_dataset = None
        self.test_dataset=None

        self.data_initialized=False

        self.len_test=None
        self.len_valid = None
        self.len_labeled=None
        self.len_unlabeled=None

        self.labeled_X_indexing_method=None
        self.labeled_y_indexing_method =None
        self.unlabeled_X_indexing_method =None
        self.unlabeled_y_indexing_method =None
        self.valid_X_indexing_method=None
        self.valid_indexing_method=None
        self.test_X_indexing_method=None
        self.test_y_indexing_method=None



        self.dataset=pd.read_csv(r'E:\桌面\LAMDA-SSL-master\graph_feature.csv')
        SemiDataset.__init__(self,pre_transform=pre_transform,transforms=transforms,transform=transform,
                             target_transform=target_transform,
                             unlabeled_transform=unlabeled_transform,test_transform=test_transform,
                             valid_transform=valid_transform,labeled_size=labeled_size,valid_size=valid_size,test_size=test_size,
                             stratified=stratified,shuffle=shuffle,random_state=random_state)
        TabularMixin.__init__(self)
        if self.default_transforms:
            self.init_default_transforms()
        self.init_dataset()

    def _init_dataset(self):
        print(self.labeled_size)
        selected_cols = ['embedding', self.column_name]
        print(self.column_name)
        df_tmp = self.dataset[selected_cols].copy()
        # 列值映射
        column_name = self.column_name
        # 替换为实际的列名
        mapping = {self.class_name[0]: 0, self.class_name[1]: 1, self.class_name[2]: 2}
        # 设置映射关系
        # 使用map函数进行映射
        df_tmp[column_name] = df_tmp[column_name].map(mapping)
        # 将df_tmp[column_name]写入文件tmp.txt
        # df_tmp[column_name].to_csv('tmp.txt', index=False)


        # 随机选取训练集
        # train_data = df_tmp.sample(frac=0.8, random_state=1)
        # # 从原始数据集中删除训练集数据
        # df_tmp = df_tmp.drop(train_data.index)
        # # 随机选取测试集
        # test_data = df_tmp.sample(frac=0.5, random_state=2)
        # # 从原始数据集中删除测试集数据
        # df_tmp = df_tmp.drop(test_data.index)
        # # 剩余数据作为验证集
        # valid_data = df_tmp
        #
        codebert_arr = df_tmp['embedding'].values
        # print(train_X.shape)
        # print(type(train_X))
        # print("train_X", train_X)
        # train_y = train_data['cvss2_confidentiality_impact'].values
        # valid_X = valid_data['codebert'].values
        # valid_y = valid_data['cvss2_I'].values

        # ##### 训练集：将bytes类型转化为ndarray类型
        X_array = np.empty((0, 768))
        for i, row in enumerate(codebert_arr):
            # s = row.decode('utf-8')
            s = row[1:-1]
            num_list = s.split(", ")
            # 将num_list中每个元素转换成float型同时输出
            num_list = [float(num) for num in num_list]
            num_array = np.array(num_list)
            num_array = num_array.reshape(1, -1)
            X_array = np.append(X_array, num_array, axis=0)
        X = X_array
        y = df_tmp[self.column_name].values.astype(np.float32)

        # X, y = self.dataset.data, self.dataset.target.astype(np.float32)

        print("X.shape, y.shape", X.shape, y.shape)
        print('X', X)
        print('y', y)
        if self.test_size is not None:
            test_X, test_y, train_X, train_y = DataSplit(X=X, y=y,
                                               size_split=self.test_size,
                                               stratified=self.stratified,
                                               shuffle=self.shuffle,
                                               random_state=self.random_state,
                                               flag='test'
                                               )
        else:
            test_X = None
            test_y = None
            train_X=X
            train_y=y

        if self.valid_size is not None:
            valid_X, valid_y, train_X, train_y = DataSplit(X=train_X, y=train_y,
                                                                   size_split=self.valid_size,
                                                                   stratified=self.stratified,
                                                                   shuffle=self.shuffle,
                                                                   random_state=self.random_state,
                                                                   flag='valid'
                                                                   )
        else:
            valid_X=None
            valid_y=None

        if self.labeled_size is not None:
            labeled_X, labeled_y, unlabeled_X, unlabeled_y = DataSplit(X=train_X,y=train_y,
                                                                   size_split=self.labeled_size,
                                                                   stratified=self.stratified,
                                                                   shuffle=self.shuffle,
                                                                   random_state=self.random_state,
                                                                   flag='labeled'
                                                                   )
        else:
            labeled_X, labeled_y=train_X,train_y
            unlabeled_X, unlabeled_y=None,None

        self.test_dataset=LabeledDataset(pre_transform=self.pre_transform,transform=self.test_transform)
        self.test_dataset.init_dataset(test_X,test_y)
        self.valid_dataset=LabeledDataset(pre_transform=self.pre_transform,transform=self.valid_transform)
        self.valid_dataset.init_dataset(valid_X,valid_y)
        self.train_dataset = TrainDataset(pre_transform=self.pre_transform,transforms=self.transforms,transform=self.transform,
                                          target_transform=self.target_transform,unlabeled_transform=self.unlabeled_transform)
        labeled_dataset=LabeledDataset(pre_transform=self.pre_transform,transforms=self.transforms,transform=self.transform,
                                          target_transform=self.target_transform)
        labeled_dataset.init_dataset(labeled_X, labeled_y)
        unlabeled_dataset=UnlabeledDataset(pre_transform=self.pre_transform,transform=self.unlabeled_transform)
        unlabeled_dataset.init_dataset(unlabeled_X, unlabeled_y)
        self.train_dataset.init_dataset(labeled_dataset=labeled_dataset,unlabeled_dataset=unlabeled_dataset)