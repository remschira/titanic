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
    #we will split the train.csv data into training and testing data
    path          = './data/'
    fileName      = 'train.csv'
    data          = pd.read_csv(path+fileName)    
    features_list = [i for i in data]

    y             = ['Survived'] #dependent variable

    #drop some of the columns from features. Such as 'Survived' which is the dependent variable.
    features      = data.drop(['Survived','PassengerId','Ticket','Cabin'],axis=1)

    #Make a new df that will contain each person's title.
    titles        = pd.DataFrame(index=features.index,columns=['Title'])#empty at the moment
    
    #This loop is where the titles are determined
    j = 0
    for i in features['Name']:
        name = HumanName(i)
        titles.loc[features.index[j]] = [name.title]
        j = j+1        

        
    #Concatenate features and features_titles
    features = pd.concat([features,titles],axis=1)



    #Split data
    X_train,X_test,y_train,y_test = train_test_split(
        features,data[y],test_size=0.33,random_state=rng)


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
    #age_miss         = X_train.loc[miss,'Age'].median()
    age_miss_old     = X_train.loc[miss_old,'Age'].median()        
    age_miss_young   = X_train.loc[miss_young,'Age'].median()
    age_mrs          = X_train.loc[mrs,'Age'].median()    


    #Fill in NaN ages with the median age for the correct Title category
    X_train.loc[master,'Age']     = X_train.loc[master,'Age'].fillna(age_master)
    X_train.loc[mr,'Age']         = X_train.loc[mr,'Age'].fillna(age_mr)
    X_train.loc[dr,'Age']         = X_train.loc[dr,'Age'].fillna(age_dr)
    #X_train.loc[miss,'Age']       = X_train.loc[miss,'Age'].fillna(age_miss)
    X_train.loc[miss_old,'Age']   = X_train.loc[miss_old,'Age'].fillna(age_miss_old)
    X_train.loc[miss_young,'Age'] = X_train.loc[miss_young,'Age'].fillna(age_miss_young)    
    X_train.loc[mrs,'Age']        = X_train.loc[mrs,'Age'].fillna(age_mrs)            


    #If have NaN for 'Embarked', replace NaN with the mode of the 'Embarked' column
    X_train['Embarked'].fillna(X_train['Embarked'].mode()[0],inplace=True) #fill nan values with mode    
    #X_train['Age'].fillna(X_train['Age'].median(),inplace=True)


    

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
    #age_miss         = X_test.loc[miss,'Age'].median()    
    age_miss_young   = X_test.loc[miss_young,'Age'].median()
    age_miss_old     = X_test.loc[miss_old,'Age'].median()    
    age_mrs          = X_test.loc[mrs,'Age'].median()    



    X_test.loc[master,'Age']     = X_test.loc[master,'Age'].fillna(age_master)
    X_test.loc[mr,'Age']         = X_test.loc[mr,'Age'].fillna(age_mr)
    X_test.loc[dr,'Age']         = X_test.loc[dr,'Age'].fillna(age_dr)
    #X_test.loc[miss,'Age']       = X_test.loc[miss,'Age'].fillna(age_miss)    
    X_test.loc[miss_old,'Age']   = X_test.loc[miss_old,'Age'].fillna(age_miss_old)
    X_test.loc[miss_young,'Age'] = X_test.loc[miss_young,'Age'].fillna(age_miss_young)        
    X_test.loc[mrs,'Age']        = X_test.loc[mrs,'Age'].fillna(age_mrs)            


    X_test['Embarked'].fillna(X_test['Embarked'].mode()[0],inplace=True) #fill nan values with mode    
    #X_test['Age'].fillna(X_test['Age'].median(),inplace=True)

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

    ##scale data
    #scaler              = preprocessing.StandardScaler()
    #scaler.fit(X_train)
    #X_train_scaled      = scaler.transform(X_train) #this converts pandas dataframe to numpy
    #X_test_scaled       = scaler.transform(X_test)
    #X_train = pd.DataFrame(X_train, index=X_train.index, columns=X_train.columns)
    #X_test  = pd.DataFrame(X_test, index=X_test.index, columns=X_test.columns)    


    c,r             = y_train.shape
    y_train_reshape = y_train.values.reshape(c,) 

    #find predictions for X_test with Gradient Boosting
    clf_gbm = GradientBoostingClassifier(n_estimators=350,max_depth=3,learning_rate=0.01,
                                         min_samples_split=4,min_samples_leaf=1)
    clf_gbm.fit(X_train,y_train_reshape)
    pred_gbm = clf_gbm.predict(X_test)


    importances   = clf_gbm.feature_importances_
    g             = sns.barplot(x=list(X_train), y=importances)

    for item in g.get_xticklabels():
        item.set_rotation(90)
    plt.show()
    print(accuracy_score(y_test,pred_gbm))





    

    file = open('./output/classifier_performance_aug.dat','w')


    file.write( '\n::::::gradient boosting::::::\n')
    lam = [1,2,3,4,5,6,10,15,20,30,50,100]

    #gradient boosting
    for l in lam:
        clf_gbm = GradientBoostingClassifier(n_estimators=350,max_depth=l,learning_rate=0.01,
                                             min_samples_split=5,min_samples_leaf=1)    
        clf_gbm.fit(X_train,y_train_reshape)
        pred_gbm = clf_gbm.predict(X_test)
        file.write( '\n \n::: with max_depth = %d :::::: \n\n' %l )        
        file.write('\naccuracy = %f \n' % accuracy_score(y_test,pred_gbm))
        file.write('confusion m. trace = %d \n' % np.trace(confusion_matrix(y_test,pred_gbm) ) )
        file.write('cross val score = %f \n' %
                   np.average(cross_val_score(clf_gbm,X_train,y_train_reshape,cv=5, scoring='accuracy'))) 
    
    file.write( '::::::logistic regression::::::\n')        
    lam = [0.001,0.01,0.1,1.,10.,100.,1000.]
    for l in lam:
        clf_logit = LogisticRegression(penalty='l2',C=l,solver='newton-cg')
        clf_logit.fit(X_train,y_train_reshape)
        pred_logit = clf_logit.predict(X_test)
        file.write( '\n \n::: with C = %f :::::: \n\n' %l )            
        file.write('\naccuracy = %f \n' % accuracy_score(y_test,pred_logit))
        mat = confusion_matrix(y_test,pred_logit)    
        file.write('confusion m. trace = %d \n' % np.trace(mat ) )
        file.write('cross val score = %f \n' %
              np.average(cross_val_score(clf_logit,X_train,y_train_reshape,cv=5, scoring='accuracy')))


        
    file.write( '\n::::::Decision Tree::::::\n')            
    lam = [2,5,8,10,15,30,50,70,100,150]
    for l in lam:
        clf_tree = DecisionTreeClassifier(max_depth=l)
        clf_tree.fit(X_train,y_train_reshape)
        pred_tree = clf_tree.predict(X_test)
        file.write( '\n \n::: with max_depth = %d :::::: \n\n' %l )
        file.write('\naccuracy = %f \n' % accuracy_score(y_test,pred_tree))
        file.write('confusion m. trace = %d \n' % np.trace(confusion_matrix(y_test,pred_tree) ) )
        file.write('cross val score = %f \n' %
                   np.average(cross_val_score(clf_tree,X_train,y_train_reshape,cv=5, scoring='accuracy')))


    file.write( '\n::::::random forest::::::\n')
    lam = [2,5,8,10,15,30,50,70,100,150]
    #n_estimators = 300 best, acc=.669,trace=353
    #default max_depth is best
    for l in lam:
        clf_rf = RandomForestClassifier(n_estimators=300,max_depth=l)    
        clf_rf.fit(X_train,y_train_reshape)
        pred_rf = clf_rf.predict(X_test)
        file.write( '\n \n::: with max_depth = %d :::::: \n\n' %l )        
        file.write('\naccuracy = %f \n' % accuracy_score(y_test,pred_rf))
        file.write('confusion m. trace = %d \n' % np.trace(confusion_matrix(y_test,pred_rf) ) )
        file.write('cross val score = %f \n' %
                   np.average(cross_val_score(clf_rf,X_train,y_train_reshape,cv=5, scoring='accuracy')))        

    file.write( '\n::::::Adaboost::::::\n')
    lam = [1,2,4,10,50,100]    
    for l in lam:
        clf_ada = AdaBoostClassifier(n_estimators=l)
        clf_ada.fit(X_train,y_train_reshape)
        pred_ada = clf_ada.predict(X_test)

        
        file.write( '\n \n::: with n_estimators = %d :::::: \n\n' %l )
        file.write('\naccuracy = %f \n' % accuracy_score(y_test,pred_ada))
        file.write('confusion m. trace = %d \n' % np.trace(confusion_matrix(y_test,pred_ada) ) )
        file.write('cross val score = %f \n' %
                   np.average(cross_val_score(clf_ada,X_train,y_train_reshape,cv=5, scoring='accuracy')))     

    file.write( '\n::::::Linear SVM::::::\n')
    lam = [0.001,0.01,0.02,0.04,0.06,0.08,0.10,1.0,5.0]    
    for l in lam:
        clf_lsvm = LinearSVC(C=l)
        clf_lsvm.fit(X_train,y_train_reshape)
        pred_lsvm = clf_lsvm.predict(X_test)
        file.write( '\n \n::: with C = %f :::::: \n\n' %l )
        file.write('\naccuracy = %f \n' % accuracy_score(y_test,pred_lsvm))
        file.write('confusion m. trace = %d \n' % np.trace(confusion_matrix(y_test,pred_lsvm) ) )
        file.write('cross val score = %f \n' %
                   np.average(cross_val_score(clf_lsvm,X_train,y_train_reshape,cv=5, scoring='accuracy'))) 

    file.write( '\n::::::SVM rbf::::::\n')
    lam = [0.001,0.01,0.02,0.04,0.06,0.08,0.10,1.0,5.0]    
    for l in lam:
        clf_kernel_svm = SVC(kernel='rbf',C=l)
        clf_kernel_svm.fit(X_train,y_train_reshape)
        pred_kernel_svm = clf_kernel_svm.predict(X_test)    
        file.write( '\n \n::: with C = %f :::::: \n\n' %l )
        file.write('\naccuracy = %f \n' % accuracy_score(y_test,pred_kernel_svm))
        file.write('confusion m. trace = %d \n' % np.trace(confusion_matrix(y_test,pred_kernel_svm) ) )
        file.write('cross val score = %f \n' %
                   np.average(cross_val_score(clf_kernel_svm,X_train,y_train_reshape,cv=5, scoring='accuracy')))
        
