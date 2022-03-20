# Name - Sagar Suman
# Roll No. 2019197
# ML Assignment - 1

# Question 1 - part 1

# importing pandas:-
import pandas as pd;
import matplotlib.pyplot as plt; 

# A) - loading the dataset using Pandas

# Note - We have added the columns names in iris.data
# Reading iris.data file with delimiter as ','-
iris_data = pd.read_csv('iris.data', delimiter=',');

#---------------------------------------------

# B) - printing the column information - 

c = 1;
for i in iris_data:
	print("Column",c,"-");	
	# Name -
	print("Column name -",i);
	# Data type -
	print("Data type -",iris_data.dtypes[i]);
	# Value range -
	print("Value range - \n min:",iris_data[i].min(),", max:",iris_data[i].max());
	# Count - 
	print("Count -",iris_data[i].count());
	if(c == 5):	
		print(iris_data[i].value_counts());	# Showing values count for each class
	print("");
	c += 1;

#--------------------------------------------------

# C) - Plotting the histogram and bar graph-

#print(iris_data.describe());

# Plotting histograms of 1st four columns 
plt.figure('Bar graph of target class')

# bar graph-
iris_data['Class'].value_counts().plot.bar(rot=0);
plt.xlabel('Classes of IRIS');
plt.ylabel('Count of classes');
plt.title('Bar graph of target class');

# histograms -
plt.figure('Histograms of continuos valued attributes');
#plotting Sepal length
plt.subplot(2,2,1);
iris_data['Sepal length'].plot.hist();
plt.title('Histogram for Sepal length');
plt.xlabel('Sepal length');
plt.ylabel('Count of plants');
#plotting Sepal width
plt.subplot(2,2,2);
iris_data['Sepal width'].plot.hist();
plt.title('Histograms for Sepal width');
plt.xlabel('Sepal width');
plt.ylabel('Count of plants');
#plotting Petal length
plt.subplot(2,2,3);
iris_data['Petal length'].plot.hist();
plt.title('Histogram for Petal length');
plt.xlabel('Petal length');
plt.ylabel('Count of plants');
#plotting Petal width
plt.subplot(2,2,4);
iris_data['Petal width'].plot.hist();
plt.title('Histograms for Petal width');
plt.xlabel('Petal width');
plt.ylabel('Count of plants');

#iris_data.hist(bins=25);
#setting spacing 
plt.tight_layout();
#Showing
plt.show();
