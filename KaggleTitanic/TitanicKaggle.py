import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import xgboost as xgb
from sklearn import svm

pd.set_option('display.max_columns', None)


data_test = pd.read_csv('test.csv')
data_train = pd.read_csv('train.csv')
data_combined = pd.concat([data_train, data_test], ignore_index=True)
data_test_id = data_test['PassengerId']


data_combined.fillna({'Survived': -1}, inplace=True)
#let's check our data. EDA

features_need = ["Survived","Pclass","Sex","SibSp","Parch","Embarked"]
fig, myplot = plt.subplots(figsize = (15,6), nrows = 2, ncols = 3)
row, col, num_cols = 0, 0, 3
for u in features_need:
    sns.barplot(x = data_combined[u].value_counts().index, y = data_combined[u].value_counts(), ax = myplot[row, col])
    col += 1
    if col == 3:
        col = 0
        row += 1
plt.subplots_adjust(hspace = 0.5)
plt.subplots_adjust(wspace = 0.3)
for v in range(2):
    for z in range(3):
        for patch in myplot[v, z].patches:
            label_x = patch.get_x() + patch.get_width()/2  # find midpoint of rectangle
            label_y = patch.get_y() + patch.get_height()/2
            myplot[v, z].text(label_x, label_y, patch.get_height(), horizontalalignment='center', verticalalignment='center')

#check nulls elements
#print(data_combined.isnull().sum()) #Age-263; Cabin-1014; Embarked-2, Fare-1.

# Let's look at each feature separately and make sure that the data was not filled in accidentally,and not intentionally
# print(data_combined.groupby(['Pclass']) ['Survived'].value_counts(normalize=True))
# print(data_combined.groupby(['Sex']) ['Pclass'].value_counts(normalize=True))

# Adding a new column in data_combined with Age=1 if there is a value and Age=0 if not
data_combined['AgeBin'] = data_combined.apply(lambda row: '0' if pd.isnull(row['Age']) else '1', axis=1)
#print(data_combined.groupby(['AgeBin', 'Pclass']) ['Survived'].value_counts(normalize=True))
#print(data_combined.groupby(['Sex','Pclass']) ['AgeBin'].value_counts())
# We see that the age is not specified for the most part in the 3rd grade (more in men than in women). We can assume
# that most likely this was not done on purpose, but simply skipped the data in the description

# To determine more precisely what age needs to be filled in, let's look at the Title of each passenger and add a new
# Title column
data_combined['Title'] = data_combined.Name.str.extract(' ([A-Za-z]+)\.', expand=False) # We get titles
frequent_titles = data_combined['Title'].value_counts()[:4].index.tolist() # We need only first 4 titles (Mr;Miss;Mrs;Master)
data_combined['Title'] = data_combined['Title'].apply(lambda row: row if row in frequent_titles else 'Other')

# Fill in the empty data in Age with the average value
sns.displot(data_combined['Age']) # Let's check the distribution of signs 'Age'
#Since the distribution is close to normal, we will calculate the average value
data_combined.fillna({'Age': data_combined.groupby('Title')['Age'].transform('mean')}, inplace = True)

g = sns.FacetGrid(data_combined, col='Survived')
g.map(plt.hist, 'Age', bins=10)

# Check Cabin with None = 1014. Creating a binary column for Cabin
data_combined['CabinBin'] = data_combined.apply(lambda row: 0 if pd.isnull(row['Cabin']) else 1, axis=1)

#print(data_combined.groupby(['CabinBin'])['Survived'].value_counts())

# Drop Cabin, because it doesn't affect the data. We have bin Cabin -> CabinBin
data_combined.drop(['Cabin'], axis=1, inplace=True)

# Check Embarked with None = 2.
#print(data_combined['Embarked'].value_counts())

#This is a categorical feature, let's replace it with a mode[0]
data_combined.fillna({'Embarked': data_combined['Embarked'].mode()[0]}, inplace=True)


# Check Fare with None = 1
sns.displot(data_combined['Fare'])
#Fare depends on the Pclass the person was in. Accordingly, we need to supplement the data with a class and an average distribution
data_combined.fillna({'Fare': data_combined.groupby('Pclass')['Fare'].transform('mean')}, inplace = True)
#Let's check our nulls elements in data
#print(data_combined.isnull().sum()) # 0 null elements

#---------------------------------------------------------------------------------------------------------------------

data_combined.drop(['Name', 'Ticket', 'AgeBin', 'Title','PassengerId'], axis = 1, inplace = True)


train_indices = range(len(data_train))
test_indices = range(len(data_train), len(data_train) + len(data_test))

train_data_new = data_combined.loc[train_indices] # Our new train_data after EDA
test_data_new = data_combined.loc[test_indices].drop(['Survived'], axis=1) # Our new test_data after EDA

# Now let's convert categorical values to numeric values using LabelEncoder
cols = ['Sex', 'Embarked']
le = LabelEncoder()
for col in cols:
    train_data_new[col] = le.fit_transform(train_data_new[col])
    test_data_new[col] = le.transform(test_data_new[col])

y = train_data_new['Survived']
X = train_data_new.drop(['Survived'], axis=1)

#Step #2 - choosing the best algorithm
KFold_Score = pd.DataFrame()
classifiers = ['Linear SVM', 'Radial SVM', 'LogisticRegression', 'RandomForestClassifier', 'AdaBoostClassifier',
    'XGBoostClassifier', 'KNeighborsClassifier','GradientBoostingClassifier']
models = [
    svm.SVC(kernel='linear'),
    svm.SVC(kernel='rbf'),
    LogisticRegression(max_iter = 1000),
    RandomForestClassifier(n_estimators=200, random_state=0),
    AdaBoostClassifier(random_state = 0, algorithm='SAMME'),
    xgb.XGBClassifier(n_estimators=100),
    KNeighborsClassifier(),
    GradientBoostingClassifier(random_state=0)
    ]

j = 0
for i in models:
    model = i
    cv = KFold(n_splits=5, random_state=0, shuffle=True)
    KFold_Score[classifiers[j]] = (cross_val_score(model, X, np.ravel(y), scoring = 'accuracy', cv=cv))
    j = j + 1

mean = pd.DataFrame(KFold_Score.mean(), index = classifiers)
KFold_Score = pd.concat([KFold_Score,mean.T])
KFold_Score.index=['Fold 1','Fold 2','Fold 3','Fold 4','Fold 5','Mean'] # 5 folds in KFold
#print(KFold_Score.T.sort_values(by=['Mean'], ascending = False)) # -> best algorithm is GradientBoostingClassifier

#Step #3 - Hyperparameters
gbc = GradientBoostingClassifier()
param_grid = {
    'n_estimators': [50, 100],  # Number of estimators (trees) in the ensemble
    'learning_rate': [0.01, 0.05],  # Learning rate
    'max_depth': [1, 5],  # Maximum depth of the trees
}

CV_gbc = GridSearchCV(estimator=gbc, param_grid=param_grid, cv= 5)
CV_gbc.fit(X,y)
#print(CV_gbc.best_params_) # best params:
#{'learning_rate': 0.05, 'max_depth': 5, 'n_estimators': 100}

# Step 4 -> fit and predict
gbcfinish = GradientBoostingClassifier(learning_rate = 0.05, max_depth = 5, n_estimators = 100).fit(X,y)
predfinish = gbcfinish.predict(test_data_new)
predfinish = np.array(list(map(np.int_, predfinish)))

submission_file = pd.DataFrame({'PassengerId': data_test_id.values, 'Survived': predfinish})
submission_file.to_csv('submission.csv', index=False)

#plt.show()


