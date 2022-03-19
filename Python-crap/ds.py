# general steps - 
# 1. import the data
# 2. clean the data
# 3. split the data(train/test)
# 4. create a model
# 5. train the model
# 6. test the model
# 7. evaluate and improve the model

# Libraries - 
# 1. numpy
# 2. pandas - read_csv(),shape,columns,values,describe(),drop()
# 3. matplotlib
# 4. scikit-learn

#Pandas -
import pandas as pd;

#Scikit learn - 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

#music_data = pd.read_csv('music.csv');
#print(music_data);

# shape attribute - 
#print("SHAPE -\n",music_data.shape);

# columns attribute - 
#print("Columns -\n",music_data.columns);
#for i in music_data.columns:
#	print(i);

# values attribute- 
#print("VALUES -\n",music_data.values);

# describe() function- 
#print("Describe -\n",music_data.describe());

# drop()function - taking out X and y:
#X = music_data.drop(columns=['genre']);
#print("X = \n",X);
#y = music_data['genre'];
#print("Y = \n",y);
 
#MODEL -> Decision tree - 
#model = DecisionTreeClassifier();
#model.fit(X,y);
#predictions = model.predict([[22,1], [22,0]]);
#print(predictions);

# Splitting the data in 80% training and 20% testing
#X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2);

#model.fit(X_train,y_train);
#predictions = model.predict(X_test);
#acc_score = accuracy_score(y_test,predictions);
#print("Accuracy score =",acc_score);

# Saving/Dumping the model:- joblib
#joblib.dump(model,'music_model.joblib');
#print("Model dumped!");

#Loading the model - 
model = joblib.load('music_model.joblib');
predictions = model.predict([[21,1]]);
print("Model loaded, prediction =",predictions);
