import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC,SVC
from nameparser import HumanName
from sklearn import preprocessing

if __name__ == '__main__':

    rng           = np.random.RandomState(42)

    #read data
    path          = './data/'
    train_file    = 'train.csv'
    test_file     = 'test.csv'    
    train_data    = pd.read_csv(path+train_file)
    test_data     = pd.read_csv(path+test_file)        
    features_list = [i for i in train_data]

    y             = ['Survived'] #dependent variable

    #drop some of the columns from features. Such as 'Survived' which is the dependent variable.
    train_features = train_data.drop(['Survived','PassengerId','Ticket','Cabin'],axis=1)
    test_features  = test_data.drop(['PassengerId','Ticket','Cabin'],axis=1)    

    #Make a new df that will contain each person's title.
    train_titles        = pd.DataFrame(index=train_features.index,columns=['Title'])#empty at the moment
    test_titles         = pd.DataFrame(index=test_features.index,columns=['Title'])#empty at the moment    
    
    #This loop is where the titles are determined
    j = 0
    for i in train_features['Name']:
        name = HumanName(i)
        train_titles.loc[train_features.index[j]] = [name.title]
        j = j+1        

    #This loop is where the titles are determined
    j = 0
    for i in test_features['Name']:
        name = HumanName(i)
        test_titles.loc[test_features.index[j]] = [name.title]
        j = j+1        

                
    #Concatenate features and features_titles
    train_features = pd.concat([train_features,train_titles],axis=1)
    test_features = pd.concat([test_features,test_titles],axis=1)    

    #X_train = train_features
    X_test  = test_features
    #y_train = train_data[y]

    
    X_train,X_blank,y_train,y_blank = train_test_split(
        train_features,train_data[y],test_size=0.33,random_state=rng)

    

    
    #produce boolean df's that contain title information
    title  = X_train['Title']
    master = (title == 'Master.')
    mr     = (title == 'Mr.') 
    dr     = (title == 'Dr.')    
    miss   = ( (title == 'Miss.') | (title == 'Ms.') |
                              (title == 'Mlle.') )
    mrs    = ( (title == 'Mrs.') | (title == 'Lady.') |
                              (title == 'Mme.') | (title == 'the Countess. of') )

    
    #Separate the 'Miss.' title into the young (who have siblings and/or parents) and
    #into old (who don't have siblings or parents)
    young      = (X_train['SibSp'] != 0) | (X_train['Parch'] !=0)
    miss_young = miss & young
    miss_old   = miss & ~young


    #Find the median age for each Title category.
    age_master       = X_train.loc[master,'Age'].median()
    age_mr           = X_train.loc[mr,'Age'].median()
    age_dr           = X_train.loc[dr,'Age'].median()
    age_miss_old     = X_train.loc[miss_old,'Age'].median()        
    age_miss_young   = X_train.loc[miss_young,'Age'].median()
    age_mrs          = X_train.loc[mrs,'Age'].median()    


    #Fill in NaN ages with the median age for the correct Title category
    X_train.loc[master,'Age']     = X_train.loc[master,'Age'].fillna(age_master)
    X_train.loc[mr,'Age']         = X_train.loc[mr,'Age'].fillna(age_mr)
    X_train.loc[dr,'Age']         = X_train.loc[dr,'Age'].fillna(age_dr)
    X_train.loc[miss_old,'Age']   = X_train.loc[miss_old,'Age'].fillna(age_miss_old)
    X_train.loc[miss_young,'Age'] = X_train.loc[miss_young,'Age'].fillna(age_miss_young)    
    X_train.loc[mrs,'Age']        = X_train.loc[mrs,'Age'].fillna(age_mrs)            


    #If have NaN for 'Embarked', replace NaN with the mode of the 'Embarked' column
    X_train['Embarked'].fillna(X_train['Embarked'].mode()[0],inplace=True) #fill nan values with mode    

    #Repeat above steps for the X_test data
    title  = X_test['Title']
    master = (title == 'Master.')
    mr     = (title == 'Mr.')
    dr     = (title == 'Dr.') 
    miss   = ( (title == 'Miss.') | (title == 'Ms.') |
                              (title == 'Mlle.') )
    mrs    = ( (title == 'Mrs.') | (title == 'Lady.') |
                              (title == 'Mme.') | (title == 'the Countess. of') )

    young      = (X_test['SibSp'] != 0) | (X_test['Parch'] !=0)
    miss_young = miss & young
    miss_old   = miss & ~young
    

    age_master       = X_test.loc[master,'Age'].median()
    age_mr           = X_test.loc[mr,'Age'].median()
    age_dr           = X_test.loc[dr,'Age'].median()
    age_miss_young   = X_test.loc[miss_young,'Age'].median()
    age_miss_old     = X_test.loc[miss_old,'Age'].median()    
    age_mrs          = X_test.loc[mrs,'Age'].median()    



    X_test.loc[master,'Age']     = X_test.loc[master,'Age'].fillna(age_master)
    X_test.loc[mr,'Age']         = X_test.loc[mr,'Age'].fillna(age_mr)
    X_test.loc[dr,'Age']         = X_test.loc[dr,'Age'].fillna(age_dr)
    X_test.loc[miss_old,'Age']   = X_test.loc[miss_old,'Age'].fillna(age_miss_old)
    X_test.loc[miss_young,'Age'] = X_test.loc[miss_young,'Age'].fillna(age_miss_young)        
    X_test.loc[mrs,'Age']        = X_test.loc[mrs,'Age'].fillna(age_mrs)            


    X_test['Embarked'].fillna(X_test['Embarked'].mode()[0],inplace=True) #fill nan values with mode
    X_test['Fare'].fillna(X_test['Fare'].median(),inplace=True)    


    #transform categorical to numeric values
    #Use the train encoding for the test data
    le = LabelEncoder()
    if X_train['Sex'].dtype=='object':
        le.fit(X_train['Sex'].values)
        X_train['Sex'] = le.transform(X_train['Sex'])
        X_test['Sex']  = le.transform(X_test['Sex'])        

    if X_train['Embarked'].dtype=='object':
        le.fit(X_train['Embarked'].values)
        X_train['Embarked'] = le.transform(X_train['Embarked'])
        X_test['Embarked']  = le.transform(X_test['Embarked'])


    #remove some columns
    X_train = X_train.drop(['Name','Title'],axis=1)
    X_test  = X_test.drop(['Name','Title'],axis=1)            


    c,r             = y_train.shape
    y_train_reshape = y_train.values.reshape(c,)

    #find predictions for X_test with Gradient Boosting
    clf_gbm = GradientBoostingClassifier(n_estimators=350,max_depth=3,learning_rate=0.01,
                                         min_samples_split=4,min_samples_leaf=1)
    clf_gbm.fit(X_train,y_train_reshape)
    pred_gbm = clf_gbm.predict(X_test)

    pred = pd.DataFrame({'PassengerId':test_data['PassengerId'],'Survived':pred_gbm})

    pred.to_csv('./output/titanic_submission2.csv',index=False)
    


    


