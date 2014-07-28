# Import common libraries
from numpy.random import rand
from numpy import ones, zeros, concatenate
from pandas import read_csv

from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

Food = read_csv('Food_Features.csv')
People = read_csv('People_Features.csv')

cTrainF = rand(len(Food)) > .5
cTestF = ~cTrainF

cTrainP = rand(len(People)) > .5
cTestP = ~cTrainP

TrainX = concatenate([People[cTrainP], Food[cTrainF]])
TestX = concatenate([People[cTestP], Food[cTestF]])

TrainY = concatenate([zeros(len(People[cTrainP])), ones(len(Food[cTrainF]))])
TestY = concatenate([zeros(len(People[cTestP])), ones(len(Food[cTestF]))])

tree = DecisionTreeClassifier(max_depth=None, min_samples_split=1, 
                              random_state=0)
tree.fit(TrainX,TrainY)
treeOut = tree.predict(TestX)
print sum(treeOut == TestY)/float(len(treeOut))

forest1 = RandomForestClassifier(n_estimators=50, max_depth=None,
                                 min_samples_split=1, random_state=0)
forest1.fit(TrainX,TrainY)
forestOut1 = forest1.predict(TestX)                             
print sum(forestOut1 == TestY)/float(len(forestOut1))
                                                                                          
forest2 = ExtraTreesClassifier(n_estimators=50, max_depth=None,
                               min_samples_split=1, random_state=0)
forest2.fit(TrainX,TrainY)
forestOut2 = forest2.predict(TestX)                             
print sum(forestOut2 == TestY)/float(len(forestOut2))

forest3 = AdaBoostClassifier(n_estimators=50, random_state=0)
forest3.fit(TrainX,TrainY)
forestOut3 = forest3.predict(TestX)                             
print sum(forestOut3 == TestY)/float(len(forestOut3))

print sum((treeOut != TestY) & (forestOut1 != TestY) & (forestOut2 != TestY) & (forestOut3 != TestY))