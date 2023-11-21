from sklearn.pipeline import Pipeline

from Co_training_codebert import Co_training_codebert
from LAMDA_SSL.Algorithm.Classification.FixMatch import FixMatch
from LAMDA_SSL.Augmentation.Vision.Cutout import Cutout
from LAMDA_SSL.Augmentation.Vision.RandAugment import RandAugment
from LAMDA_SSL.Augmentation.Vision.RandomCrop import RandomCrop
from LAMDA_SSL.Augmentation.Vision.RandomHorizontalFlip import RandomHorizontalFlip
from LAMDA_SSL.Evaluation.Classifier.Accuracy import Accuracy
from LAMDA_SSL.Evaluation.Classifier.Precision import Precision
from LAMDA_SSL.Evaluation.Classifier.Recall import Recall
from LAMDA_SSL.Evaluation.Classifier.F1 import F1
from LAMDA_SSL.Evaluation.Classifier.AUC import AUC
from LAMDA_SSL.Evaluation.Classifier.Confusion_Matrix import Confusion_Matrix
from LAMDA_SSL.Network.FT_Transformer import FT_Transformer
import numpy as np
from LAMDA_SSL.Opitimizer.Adam import Adam
from LAMDA_SSL.Sampler.RandomSampler import RandomSampler
from LAMDA_SSL.Dataset.LabeledDataset import LabeledDataset
from LAMDA_SSL.Dataset.UnlabeledDataset import UnlabeledDataset
from LAMDA_SSL.Transform.ToTensor import ToTensor
from LAMDA_SSL.Augmentation.Tabular.Noise import Noise
from time import time

column_names = ['cvss2_serverity', 'cvss2_AV', 'cvss2_AC', 'cvss2_C', 'cvss2_I', 'cvss2_A']
class_names = ['HLM', 'ALN', 'HLM', 'CNP', 'CNP', 'CNP']
for i in range(1, 7):
    class_name = class_names[i - 1]
    column_name = column_names[i - 1]
    filepath = './GraphFeaTureResult/cvss2_' + column_name + '_0.3(T)' + class_name + '.txt'

    start_time = time()
    file = open(filepath, "w")

    dataset=Co_training_codebert(test_size=0.1,labeled_size=0.3,stratified=True,shuffle=True,random_state=42,default_transforms=True, class_name=class_name, column_name=column_name)
    labeled_X=dataset.labeled_X
    labeled_y=dataset.labeled_y
    unlabeled_X=dataset.unlabeled_X
    unlabeled_y=dataset.unlabeled_y
    test_X=dataset.test_X
    test_y=dataset.test_y

    # Pre_transform
    pre_transform=dataset.pre_transform
    pre_transform.fit(np.vstack([labeled_X, unlabeled_X]))

    labeled_X=pre_transform.transform(labeled_X)
    unlabeled_X=pre_transform.transform(unlabeled_X)
    test_X=pre_transform.transform(test_X)

    labeled_dataset=LabeledDataset(transform=ToTensor())
    unlabeled_dataset=UnlabeledDataset(transform=ToTensor())
    test_dataset=UnlabeledDataset(transform=ToTensor())

    labeled_sampler=RandomSampler(replacement=True,num_samples=64*(10000))

    evaluation={
        'accuracy':Accuracy(),
        'precision':Precision(average='macro'),
        'Recall':Recall(average='macro'),
        'F1':F1(average='macro'),
        'AUC':AUC(multi_class='ovo'),
        'Confusion_matrix':Confusion_Matrix(normalize='true')
    }
    weak_augmentation1=Pipeline([('RandomHorizontalFlip',RandomHorizontalFlip()),
                                  ('RandomCrop',RandomCrop(padding=0.125,padding_mode='reflect')),
                                  ])
    strong_augmentation1=Pipeline([('RandomHorizontalFlip',RandomHorizontalFlip()),
                                  ('RandomCrop',RandomCrop(padding=0.125,padding_mode='reflect')),
                                  ('RandAugment',RandAugment(n=2,m=10,num_bins=10)),
                                  ('Cutout',Cutout(v=0.5,fill=(127, 127, 127))),
                                  ])
    weak_augmentation2=Noise(0.1)
    strong_augmentation2=Noise(0.2)
    augmentation={
        'weak_augmentation1':weak_augmentation1,
        'strong_augmentation1':strong_augmentation1,
        'weak_augmentation2':weak_augmentation2,
        'strong_augmentation2':strong_augmentation2
    }

    model=FixMatch(labeled_dataset=labeled_dataset, unlabeled_dataset=unlabeled_dataset,
                    test_dataset=test_dataset, device='cuda:0', augmentation=augmentation,
                    network=FT_Transformer(dim_in=labeled_X.shape[1], num_classes=3),num_it_epoch=10000,labeled_sampler=labeled_sampler, optimizer=Adam(lr=1e-4),
                    scheduler=None, weight_decay=1e-5,verbose=True)

    model.fit(X=labeled_X,y=labeled_y,unlabeled_X=unlabeled_X)
    performance=model.evaluate(X=test_X,y=test_y)
    result=model.y_pred
    print(result,file=file)

    print(performance,file=file)

    # 计算mcc
    from sklearn.metrics import matthews_corrcoef
    mcc = matthews_corrcoef(test_y, result)
    print('mcc\t', mcc, file=file)

    end_time = time()
    print('time:\t', end_time - start_time, file=file)

    # 保存模型
    import pickle
    with open('./GraphFeaTureResult/cvss2_' + column_name + '_0.3(T)' + class_name + '.pkl', 'wb') as f:
        pickle.dump(model, f)
    file.close()

