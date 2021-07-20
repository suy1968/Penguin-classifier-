# import all the libraries/modules
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

penguins = pd.read_csv('penguins_cleaned.csv')


# ordinal feature encoding
df = penguins.copy()
target = 'species'
encode = ['sex','island']


for col in encode:
    dummy = pd.get_dummies(df[col],prefix = col)
    df = pd.concat([df,dummy], axis = 1)
    del df[col]
    
target_mapper = {'Adelie':0, 'Chinstrap':1,'Gentoo':2}
def target_encode(val):
    return target_mapper[val]

df['species']  = df['species'].apply(target_encode)


# separating x and y
X = df.drop('species', axis = 1)
Y = df['species']


# building random forest classifier
clf = RandomForestClassifier()
clf.fit(X,Y)

# saving the model
pickle.dump(clf,open('penguin_clf.pkl','wb'))



