import numpy                       as np
import pandas                      as pd
import matplotlib.pyplot           as plt
import seaborn                     as sns; sns.set()
from sklearn.model_selection       import train_test_split,cross_val_score,cross_val_predict
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.metrics               import confusion_matrix, accuracy_score
from sklearn.preprocessing         import LabelEncoder
from sklearn.ensemble              import RandomForestClassifier, AdaBoostClassifier,GradientBoostingClassifier
from sklearn.tree                  import DecisionTreeClassifier
from sklearn.svm                   import LinearSVC,SVC
from nameparser                    import HumanName
from sklearn                       import preprocessing


rng                = np.random.RandomState(42)


trainDataFile      = './data/train.csv'
testDataFile       = './data/test.csv' 
train_df           = pd.read_csv(trainDataFile)
test_df            = pd.read_csv(testDataFile)

#dataframe containing IDs for test data homes
test_IDs_df        = pd.DataFrame({'PassengerId':test_df['PassengerId']}) 

titles        = pd.DataFrame(index=train_df.index,columns=['Title'])#empty at the moment
titles_test   = pd.DataFrame(index=test_df.index,columns=['Title'])#empty at the moment

#This loop is where the titles are determined
j = 0
for i in train_df['Name']:
    name = HumanName(i)
    titles.loc[train_df.index[j]] = [name.title]
    j = j+1
j = 0
for i in test_df['Name']:
    name = HumanName(i)
    titles_test.loc[test_df.index[j]] = [name.title]
    j = j+1

    
#Concatenate features and features_titles
train_df = pd.concat([train_df,titles],axis=1)
test_df  = pd.concat([test_df,titles_test],axis=1)


#drop some of the columns from features.
train_df      = train_df.drop(['PassengerId','Ticket','Cabin'],axis=1)
test_df       = test_df.drop(['PassengerId','Ticket','Cabin'],axis=1)

train_y       = pd.DataFrame({'Survived':train_df['Survived']}) 
train_X       = train_df.drop(['Survived'], axis=1)


dfs = [train_X,test_df]

for df in dfs:

    #produce boolean df's that contain title information
    title  = df['Title']

    master = (title == 'Master.')
    mr     = (title == 'Mr.') 
    dr     = (title == 'Dr.')    
    miss   = ( (title == 'Miss.') | (title == 'Ms.') |
               (title == 'Mlle.') )
    mrs    = ( (title == 'Mrs.') | (title == 'Lady.') |
               (title == 'Mme.') | (title == 'the Countess. of') )

    #CONSIDER MAKING YOUNG ONLY DEPENDENT ON IF SHE HAS PARENTS
    #Separate the 'Miss.' title into the young (who have siblings and/or parents) and
    #into old (who don't have siblings or parents)
    young      = (df['SibSp'] != 0) | (df['Parch'] !=0)
    miss_young = miss & young
    miss_old   = miss & ~young

    #Find the median age for each Title category.
    age_master       = df.loc[master,'Age'].median()
    age_mr           = df.loc[mr,'Age'].median()
    age_dr           = df.loc[dr,'Age'].median()
    age_miss_old     = df.loc[miss_old,'Age'].median()
    age_miss_young   = df.loc[miss_young,'Age'].median()
    age_mrs          = df.loc[mrs,'Age'].median()    


    #Fill in NaN ages with the median age for the correct Title category

    df.loc[master,'Age']     = df.loc[master,'Age'].fillna(age_master)
    df.loc[mr,'Age']         = df.loc[mr,'Age'].fillna(age_mr)
    df.loc[dr,'Age']         = df.loc[dr,'Age'].fillna(age_dr)
    df.loc[miss_old,'Age']   = df.loc[miss_old,'Age'].fillna(age_miss_old)
    df.loc[miss_young,'Age'] = df.loc[miss_young,'Age'].fillna(age_miss_young)    
    df.loc[mrs,'Age']        = df.loc[mrs,'Age'].fillna(age_mrs)

    

    df['Embarked'].fillna(df['Embarked'].mode()[0],inplace=True) 
    df['Fare'].fillna(df['Fare'].median(),inplace=True) 
    #df['Age'].fillna(df['Age'].median(),inplace=True) 

train_X = train_X.drop(['Name'],axis=1)
train_X = pd.get_dummies(train_X, columns=['Sex'])
train_X = pd.get_dummies(train_X, columns=['Embarked'])
train_X = pd.get_dummies(train_X, columns=['Title'])        

test_df = test_df.drop(['Name'],axis=1)
test_df = pd.get_dummies(test_df, columns=['Sex'])
test_df = pd.get_dummies(test_df, columns=['Embarked'])
test_df = pd.get_dummies(test_df, columns=['Title'])        


#test for NaNs
#for i in list(test_df.columns.values):
#    print(i)
#    if test_df[i].isnull().any() == True:
#            print(i,test_df[i].isnull().any())


#print('###before###')
#print(train_X.info())
#print(test_df.info())


test_df, train_X = test_df.align(train_X,join='inner', axis=1)

#train_X = train_X.drop(['Title_Rev.'],axis=1)
#test_df = test_df.drop(['Title_Rev.'],axis=1)

#train_X = train_X.drop(['Title_Dr.'],axis=1)
#test_df = test_df.drop(['Title_Dr.'],axis=1)


#print('###after###')
#print(train_X.info())
#print(test_df.info())

#scale data
scaler              = preprocessing.StandardScaler()
scaler.fit(train_X)
np_train_X          = scaler.transform(train_X.values) #this converts pandas dataframe to numpy
np_test_df          = scaler.transform(test_df.values)

train_X   = pd.DataFrame(np_train_X, index=train_X.index, columns=train_X.columns)
test_df   = pd.DataFrame(np_test_df, index=test_df.index, columns=test_df.columns)


c,r             = train_y.shape
train_y_reshape = train_y.values.reshape(c,) 


#find predictions for test_df with Gradient Boosting
clf_gbm = GradientBoostingClassifier(n_estimators=350,max_depth=3,learning_rate=0.01,
                                         min_samples_split=4,min_samples_leaf=1)
clf_gbm.fit(train_X,train_y_reshape)
pred_val_gbm = clf_gbm.predict(test_df)



importances   = clf_gbm.feature_importances_
g             = sns.barplot(x=list(train_X), y=importances)

for item in g.get_xticklabels():
    item.set_rotation(90)

plt.show()


'''
file = open('./output/classifier_performance_sept.dat','w')

file.write( '::::::gradient boosting::::::\n')
lam = [1,2,3,4,5,6,10,15,20,30,50,100]
#gradient boosting
for l in lam:
    file.write('parameter= %f\n' % l)
    clf_gbm = GradientBoostingClassifier(n_estimators=350,max_depth=l,learning_rate=0.01,
                                             min_samples_split=5,min_samples_leaf=1)    
    clf_gbm.fit(train_X,train_y_reshape)
    pred_y_gbm = cross_val_predict(clf_gbm,train_X,train_y_reshape,cv=5)
    m = confusion_matrix(train_y_reshape,pred_y_gbm)
    file.write('confusion matrix elements\n')
    for line in m:        
        np.savetxt(file, line, fmt='%.2f')
    file.write('cross val score = %f \n' %
               np.average(cross_val_score(clf_gbm,train_X,train_y_reshape,cv=5, scoring='accuracy'))) 


file.write( '\n::::::logistic regression::::::\n')        
lam = [0.001,0.01,0.1,1.,10.,100.,1000.]
for l in lam:
    file.write('parameter= %f\n' % l)    
    clf_logit = LogisticRegression(penalty='l2',C=l,solver='newton-cg')
    clf_logit.fit(train_X,train_y_reshape)
    pred_y_logit = cross_val_predict(clf_logit,train_X,train_y_reshape,cv=5)
    m = confusion_matrix(train_y_reshape,pred_y_logit)
    file.write('confusion matrix elements\n')
    for line in m:        
        np.savetxt(file, line, fmt='%.2f')
    file.write('cross val score = %f \n' %
               np.average(cross_val_score(clf_logit,train_X,train_y_reshape,cv=5, scoring='accuracy')))

file.write( '\n::::::random forest::::::\n')
lam = [2,5,8,10,15,30,50,70,100,150]
#n_estimators = 300 best, acc=.669,trace=353
#default max_depth is best
for l in lam:
    file.write('parameter= %f\n' % l)    
    clf_rf = RandomForestClassifier(n_estimators=300,max_depth=l)    
    clf_rf.fit(train_X,train_y_reshape)
    pred_y_rf = cross_val_predict(clf_rf,train_X,train_y_reshape,cv=5)    
    m = confusion_matrix(train_y_reshape,pred_y_rf)
    file.write('confusion matrix elements\n')
    for line in m:        
        np.savetxt(file, line, fmt='%.2f')        
    file.write('cross val score = %f \n' %
               np.average(cross_val_score(clf_rf,train_X,train_y_reshape,cv=5, scoring='accuracy')))        


file.write( '\n::::::Linear SVM::::::\n')
lam = [0.001,0.01,0.02,0.04,0.06,0.08,0.10,1.0,5.0]    
for l in lam:
    file.write('parameter= %f\n' % l)    
    clf_lsvm = LinearSVC(C=l)
    clf_lsvm.fit(train_X,train_y_reshape)
    pred_y_lsvm = cross_val_predict(clf_lsvm,train_X,train_y_reshape,cv=5)        
    m = confusion_matrix(train_y_reshape,pred_y_lsvm)
    file.write('confusion matrix elements\n')
    for line in m:        
        np.savetxt(file, line, fmt='%.2f')                
    file.write('cross val score = %f \n' %
               np.average(cross_val_score(clf_lsvm,train_X,train_y_reshape,cv=5, scoring='accuracy'))) 


'''
clf_gbm = GradientBoostingClassifier(n_estimators=350,max_depth=3,learning_rate=0.01,
                                             min_samples_split=4,min_samples_leaf=1)    
clf_gbm.fit(train_X,train_y_reshape)
pred_test_gbm = clf_gbm.predict(test_df)


clf_logit = LogisticRegression(penalty='l2',C=1,solver='newton-cg')
clf_logit.fit(train_X,train_y_reshape)
pred_test_logit = clf_logit.predict(test_df)

clf_rf = RandomForestClassifier(n_estimators=300,max_depth=5)    
clf_rf.fit(train_X,train_y_reshape)
pred_test_rf = clf_rf.predict(test_df)

clf_lsvm = LinearSVC(C=0.02)
clf_lsvm.fit(train_X,train_y_reshape)
pred_test_lsvm = clf_lsvm.predict(test_df)

pred_test_ave    = (1./4.0) * ( pred_test_gbm + pred_test_logit + pred_test_rf + pred_test_lsvm)


#indexes = pred_test_ave < 0.5
#pred_test_ave[indexes] = 0
#pred_test_ave[~indexes] = 1


#pred = pd.DataFrame({'PassengerId':test_IDs_df['PassengerId'],'Survived':pred_test_gbm})

#pred.to_csv('./output/titanic_submission_gbm.csv',index=False)

