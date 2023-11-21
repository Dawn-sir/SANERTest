from LAMDA_SSL.Algorithm.Classification.Assemble import Assemble
from LAMDA_SSL.Evaluation.Classifier.Accuracy import Accuracy
from LAMDA_SSL.Evaluation.Classifier.Precision import Precision
from LAMDA_SSL.Evaluation.Classifier.Recall import Recall
from LAMDA_SSL.Evaluation.Classifier.F1 import F1
from LAMDA_SSL.Evaluation.Classifier.AUC import AUC
from LAMDA_SSL.Evaluation.Classifier.Confusion_Matrix import Confusion_Matrix
from sklearn.svm import SVC
import numpy as np
from codebert import codebert

file = open("./GraphFeaTureResult/cvss2_AU_0.3(T)_NS.txt", "w")

dataset=codebert(test_size=0.1,labeled_size=0.3,stratified=True,shuffle=True,random_state=42,default_transforms=True)

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

evaluation={
    'accuracy':Accuracy(),
    'precision':Precision(average='macro'),
    'Recall':Recall(average='macro'),
    'F1':F1(average='macro'),
    'AUC':AUC(multi_class='ovo'),
    'Confusion_matrix':Confusion_Matrix(normalize='true')
}

# Base estimater
SVM=SVC(probability=True)

model=Assemble(T=100,base_estimator=SVM,evaluation=evaluation,verbose=True,file=file)

model.fit(X=labeled_X,y=labeled_y,unlabeled_X=unlabeled_X)

performance=model.evaluate(X=test_X,y=test_y)

result=model.y_pred

print(result,file=file)

print(performance,file=file)

# 计算mcc
from sklearn.metrics import matthews_corrcoef
mcc=matthews_corrcoef(test_y,result)
print(mcc,file=file)

