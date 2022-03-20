# Name - Sagar Suman
# Roll No. 2019197
# ML Assignment 1

# Question 2 :

# Note :-  Please see at the end of this file for functions calls. 
# Each parts of question 2 can be called by thier respective functions.

#-------------------------------------------------------------------------
# importing libraries -

import pandas as pd;
import matplotlib.pyplot as plt;
from sklearn.model_selection import train_test_split; # For train_test_split
from sklearn.model_selection import KFold;		# For Kfold
import numpy as np;

# Note - We have added column names in Dataset.data file.

# Reading the dataset -
abalone_data = pd.read_csv('Dataset.data', delimiter =' ');


# Taking out X and y-
X = abalone_data.drop(columns = ['Rings']);
y = abalone_data['Rings'];

# Converting 1st column of X to integer
# M maps to 0
# F maps to 1
# I maps to 2
X['Sex'] = X['Sex'].replace(['M','F','I'],[0,1.0,2.0]);

# Splitting the dataset into 90%(training+validation) and 10%(test)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.10);


#-----------------------------------------------------------------------------

# Various functions definition follows in order :- 

#--------------------------------------------------

# Function for Data visualization -
def feature_visualize(X_train):
	X_train.hist();
	plt.tight_layout();
	plt.show();

# Function for normalizing training-
def normalize_train(X_train):
	c = 0;
	bounds_list = [];			# this will contains mini and maxi
	for i in X_train.columns:
		mini = X_train[i].min();		#finding minimum
		maxi = X_train[i].max();		# finding maximum
		bounds_list.append([mini,maxi]);
		lst = np.asarray(X_train[i]);
		lst = (lst - mini)/maxi;		# normalizing
		X_train = X_train.drop(columns=[i]);
		X_train.insert(c,i,lst);
		c += 1;
	return X_train,bounds_list;


# Function for normalizing test/val set according to train set
def normalize_test(X_test,bounds_list):
	c = 0;
	for i in X_test.columns:
		mini = bounds_list[c][0];		#minimum which is used to normalize x_train
		maxi = bounds_list[c][1];		#maximum which is used to normalize x_train
		lst = np.asarray(X_test[i]);
		lst = (lst - mini)/maxi;		# normalizing x_test in same way x_
		X_test = X_test.drop(columns=[i]);
		X_test.insert(c,i,lst);
		c += 1;
	return X_test;


# Defining cost funcion -
def rmse_cost(X_train,y_train,w):
	#RMSE - Mean square error -
	res = (X_train).dot(w); # X.w
	res = (res - y_train);  # X.w - y
	res = res.dot(res);		# sum of square(X.w - y)
	res = (res/len(X_train));	# Taking mean
	return res**0.5;		# returning square root


# Function for returning regularization term for gradient decsent
def regularize(order,w,reg_param):
	if (order == 0):  #no regularization
		return 0;	
	if (order == 1):  #L1- lasso regularization
		if(w<0):
			return -reg_param;
		elif(w>0):
			return reg_param;
		else:
			return 0;
	if	(order == 2): # L2 - ridge regularization
		return 2*reg_param*w;


# Gradient descent - 
# parameters are- X, y, w(parameters), learning_rate, iterations,
# 				reg(type of regularization), reg_param(lambda)
# for linear regression : reg = 0
# for linear regression with L1 regularization : reg = 1
# for linear regression with L2 regularization : reg = 2

# Default: reg = 0
def gradient_descent(X_train,y_train,w,learning_rate,iterations,reg = 0,reg_param=0):
	i = 1;
	lst = list(range(iterations));
	# Runing for specified iterations
	while(i <= iterations):
		#Storing rmse values
		lst[i-1] = rmse_cost(X_train,y_train,w);
		res = X_train.dot(w); # X.w
		res = res - y_train;  # X.w - y
		
		j = 0;	
		for col in X_train.columns:	# updating all w -
			res2 = res.dot(X_train[col]) # sum((X.w - y).X[i])
			#updating regularize
			w[j] = w[j] - (((learning_rate)/len(X_train))*(res2)) + regularize(reg,w[j],reg_param);
			j += 1;
		i +=1;
	#returning rmse costs
	return lst;


# Defining function for linear regression and plotting-
def linear_regression(X_train,y_train,learning_rate,iterations,foldno = 0,reg=0,reg_param=0):
	
	# Inserting columns of 1, making sure x0 = 1
	if not('x0' in X_train):
		X_train.insert(0,'x0',1);	# So in total have 9 features
	
	w = np.asarray([1,1,1,1,1,1,1,1,1.0]); # 9 parameters initialized
	
	# Training using gradient descent
	rmse = gradient_descent(X_train,y_train,w,learning_rate,iterations,reg,reg_param);
	if(foldno != 0):
		# Plotting - 
		plt.subplot(2,3,foldno);
		plt.plot(range(0,iterations),rmse);
		plt.xlabel('Iterations');
		plt.ylabel('RMSE value');
		plt.title(f"For fold {foldno}");
	# Returing trained parameters
	return w;

#--------------------------------------------------------------------------------------

# Function for Q2 part1  - only Linear regression
def part1(X_train,y_train):
	# Using K-fold -
	kfold = KFold(n_splits = 5);	# 5 folds split
	# Creating figure for plotting 
	plt.figure("Iterations vs RMSE graph for linear regression");
	plt.suptitle("Iterations vs RMSE graph for different folds for linear regression");

	# Iterating over 5 folds -
	c =1;
	tot_err = 0;
	for train,test in kfold.split(X_train):
		# Getting training and validation sets
		X_fold_train = X_train.iloc[train];
		y_fold_train = y_train.iloc[train];
		X_fold_test = X_train.iloc[test];
		y_fold_test = y_train.iloc[test];
		
		#Normalizing X_fold_train -
		X_fold_train,bounds_list = normalize_train(X_fold_train); 
		#Normalizing X_fold_test according to bounds of X_fold_train
		X_fold_test = normalize_test(X_fold_test,bounds_list);
		
		# Linear regression on training set- 
		# tuning learning rate and no.of iterations - 
		learning_rate = 0.2;
		iterations = 200;
		w = linear_regression(X_fold_train,y_fold_train,learning_rate,iterations,c);
	
		# Calculating RMSE on validation set -
		# adding x0 = 1
		X_fold_test.insert(0,'x0',1);
		rmse = rmse_cost(X_fold_test,y_fold_test,w);
		tot_err += rmse;
		print(f"For Linear regression, RMSE value on fold {c} validation set is: {rmse}");
		c += 1;

	# Reporting total avg error across 5 folds
	tot_err = tot_err/5;
	print(f"For Linear regression, Average RMSE value of all folds is: {tot_err}\n");

	# Showing graph - 
	plt.tight_layout();
	plt.show();


#------------------------------------------------------------------------------------


# Function for Q2 part2  - Linear regression with L1
def part2_L1(X_train,y_train):
	# Using K-fold -
	kfold = KFold(n_splits = 5);	# 5 folds split
	# Creating figure for plotting 
	plt.figure("Iterations vs RMSE graph for linear regression with L1 regularization");
	plt.suptitle("Iterations vs RMSE graph for different folds for linear regression with L1 regularization");

	# Iterating over 5 folds -
	c =1;
	tot_err = 0;
	for train,test in kfold.split(X_train):
		# Getting training and validation sets
		X_fold_train = X_train.iloc[train];
		y_fold_train = y_train.iloc[train];
		X_fold_test = X_train.iloc[test];
		y_fold_test = y_train.iloc[test];
	
		#Normalizing X_fold_train -
		X_fold_train,bounds_list = normalize_train(X_fold_train); 
		#Normalizing X_fold_test according to bounds of X_fold_train
		X_fold_test = normalize_test(X_fold_test,bounds_list);
		
		# Linear regression on training set-
		# Passing reg = 1(for L1) and reg_param(for lambda)
		# tuning learning rate, no.of iterations and reg_param- 
		learning_rate = 0.2;
		iterations = 200;
		reg_param = 0.01;
		w = linear_regression(X_fold_train,y_fold_train,learning_rate,iterations,c,reg=1,reg_param=reg_param);
	
		# Calculating RMSE on validation set -
		# adding x0 = 1
		X_fold_test.insert(0,'x0',1);
		rmse = rmse_cost(X_fold_test,y_fold_test,w);
		tot_err += rmse;
		print(f"For Linear regression with L1 reg, RMSE value on fold {c} val set is: {rmse}");
		c += 1;

	# Reporting total avg error across 5 folds
	tot_err = tot_err/5;
	print(f"For Linear regression with L1 reg, Average RMSE value of all folds is: {tot_err}\n");

	# Showing graph - 
	plt.tight_layout();
	plt.show();


#-----------------------------------------------------------------------------


# Function for Q2 part2  - Linear regression with L2
def part2_L2(X_train,y_train):
	# Using K-fold -
	kfold = KFold(n_splits = 5);	# 5 folds split
	# Creating figure for plotting 
	plt.figure("Iterations vs RMSE graph for linear regression with L2 regularization");
	plt.suptitle("Iterations vs RMSE graph for different folds for linear regression with L2 regularization");

	# Iterating over 5 folds -
	c =1;
	tot_err = 0;
	for train,test in kfold.split(X_train):
		# Getting training and validation sets
		X_fold_train = X_train.iloc[train];
		y_fold_train = y_train.iloc[train];
		X_fold_test = X_train.iloc[test];
		y_fold_test = y_train.iloc[test];
		
		#Normalizing X_fold_train -
		X_fold_train,bounds_list = normalize_train(X_fold_train); 
		#Normalizing X_fold_test according to bounds of X_fold_train
		X_fold_test = normalize_test(X_fold_test,bounds_list);
		
		# Linear regression on training set-
		# Passing reg = 2(for L2) and reg_param(for lambda)
		# tuning learning rate, no.of iterations and reg_param- 
		learning_rate = 0.2;
		iterations = 200;
		reg_param = 0.005;
		w = linear_regression(X_fold_train,y_fold_train,learning_rate,iterations,c,reg=2,reg_param=reg_param);	
	
		# Calculating RMSE on validation set -
		# adding x0 = 1
		X_fold_test.insert(0,'x0',1);
		rmse = rmse_cost(X_fold_test,y_fold_test,w);
		tot_err += rmse;
		print(f"For Linear regression with L2 reg, RMSE value on fold {c} val set is: {rmse}");
		c += 1;

	# Reporting total avg error across 5 folds
	tot_err = tot_err/5;
	print(f"For Linear regression with L2 reg, Average RMSE value of all folds is: {tot_err}\n");

	# Showing graph - 
	plt.tight_layout();
	plt.show();

#-----------------------------------------------------------------------------

# Function for Q2 part 3 : Testing on test set 
def part3(X_train,y_train,X_test,y_test):
	
	# As we now have right set of paramters,
	# Here we will use 90% of data (train + val) for training.
	# And 10% for testing for different models.
	
	#Normalizing X_train -
	X_train,bounds_list = normalize_train(X_train); 
	#Normalizing X_test according to bounds of X_fold_train
	X_test = normalize_test(X_test,bounds_list);
		

	# Only Linear regression-
	# learning rate and no.of iterations - which we have got in part1
	learning_rate = 0.2;
	iterations = 200;
	w = linear_regression(X_train,y_train,learning_rate,iterations);
	# Calculating RMSE on testing  set -
	# adding x0 = 1
	X_test.insert(0,'x0',1);
	rmse = rmse_cost(X_test,y_test,w);
	print(f"For Only Linear regression, RMSE value on testing set is: {rmse}");


	# Linear regression + L1 -
	# Passing reg = 1(for L1) and reg_param(for lambda)
	# learning rate, no.of iterations and reg_param - which we have got in part2 
	learning_rate = 0.2;
	iterations = 200;
	reg_param = 0.01;
	w = linear_regression(X_train,y_train,learning_rate,iterations,reg=1,reg_param=reg_param);
	# Calculating RMSE on testing set -
	rmse = rmse_cost(X_test,y_test,w);
	print(f"For Linear regression + L1, RMSE value on testing set is: {rmse}");


	# Linear regression + L2 -
	# Passing reg = 2(for L2) and reg_param(for lambda)
	# learning rate, no.of iterations and reg_param - which we have got in part2 
	learning_rate = 0.2;
	iterations = 200;
	reg_param = 0.005;
	w = linear_regression(X_train,y_train,learning_rate,iterations,reg=2,reg_param=reg_param);
	# Calculating RMSE on testing set -
	rmse = rmse_cost(X_test,y_test,w);
	print(f"For Linear regression + L2, RMSE value on testing set is: {rmse}\n");
	X_test.drop(columns=['x0']);
	X_train.drop(columns=['x0']);


#--------------------------------------------------------------------------------


# Q2 part 4 - Using sklearn linear regression

# importing Linear regression from sklearn 
from sklearn.linear_model import LinearRegression;  
from sklearn.linear_model import Ridge;
from sklearn.linear_model import Lasso;

# Defining function to evaluate avg cross validation score
# across LR, LR+L1 , LR+L2, where LR - linear regresion
def evaluate_model_train(model,model_name,X_train,y_train):
	# Using kfold-
	kfold = KFold(n_splits = 5);
	# Getting 5 splits -
	c = 1;
	tot_err = 0;
	for train,test in kfold.split(X_train):
		X_fold_train = X_train.iloc[train];
		y_fold_train = y_train.iloc[train];
		X_fold_test = X_train.iloc[test];
		y_fold_test = y_train.iloc[test];
	
		#Normalizing X_fold_train -
		X_fold_train,bounds_list = normalize_train(X_fold_train); 
		#Normalizing X_fold_test according to bounds of X_fold_train
		X_fold_test = normalize_test(X_fold_test,bounds_list);
	
		# Using given model - 
		#fitting
		model.fit(X_fold_train,y_fold_train);
		# predicting
		y_fold_hat = model.predict(X_fold_test);	

		# Calculating rmse - 
		rmse = y_fold_hat - y_fold_test; #subtracting
		rmse = rmse.dot(rmse);		# Sum of square erros
		rmse = rmse/len(y_fold_hat);# MSE
		rmse = rmse**0.5;			# RMSE
		print(f"By using sklearn {model_name} on fold {c} we get RMSE: {rmse}");
		tot_err += rmse;
		c += 1;

	tot_err = tot_err/5;
	print(f"By using sklearn {model_name}, AVERAGE RMSE on all folds: {tot_err}\n");


def evaluate_model_test(model,model_name,X_train,y_train,X_test,y_test):
	# Now using 90% of data (train + val) for training,
	# and 10 % for testing -
	
	#Normalizing X_train -
	X_train,bounds_list = normalize_train(X_train); 
	#Normalizing X_test according to bounds of X_fold_train
	X_test = normalize_test(X_test,bounds_list);
	
	# Using given model - 
	# fitting 
	model.fit(X_train,y_train);
	# predicting 
	y_hat = model.predict(X_test);
	# Calculating rmse - 
	rmse = y_hat - y_test; #subtracting
	rmse = rmse.dot(rmse);		# Sum of square erros
	rmse = rmse/len(y_hat);# MSE
	rmse = rmse**0.5;			# RMSE
	print(f"By using sklearn {model_name} on testing set,we get RMSE: {rmse}");


def part4(X_train,y_train,X_test,y_test):
	# Training and tuning paramenters-
	lasso_param = 0.01;
	ridge_param = 0.05;
	evaluate_model_train(LinearRegression(),'Linear Regression',X_train,y_train);
	evaluate_model_train(Lasso(alpha = lasso_param),'Lasso Regression(L1)',X_train,y_train);
	evaluate_model_train(Ridge(alpha = ridge_param),'Ridge Regression(L2)',X_train,y_train);
	# FInal testing
	evaluate_model_test(LinearRegression(),'Linear Regression',X_train,y_train,X_test,y_test);
	evaluate_model_test(Lasso(alpha = lasso_param),'Lasso Regression(L1)',X_train,y_train,X_test,y_test);
	evaluate_model_test(Ridge(alpha = ridge_param),'Ridge Regression(L2)',X_train,y_train,X_test,y_test);
	print("\n");

#---------------------------------------------------------------------------------


# Q2 part 5 - closed form solution 

def part5(X_train,y_train):

	kfold = KFold(n_splits = 5); # 5 splits
	c = 1;
	tot_err = 0;
	# For each split -
	for train, test in kfold.split(X_train):
		X_fold_train = X_train.iloc[train];
		y_fold_train = y_train.iloc[train];
		X_fold_test = X_train.iloc[test];
		y_fold_test = y_train.iloc[test];
	
		#Normalizing X_fold_train -
		X_fold_train,bounds_list = normalize_train(X_fold_train); 
		#Normalizing X_fold_test according to bounds of X_fold_train
		X_fold_test = normalize_test(X_fold_test,bounds_list);

		# Using formula : w = inverse(X).y 
		# As X is rectangular matrix- 
		# So, inverse(X) = (inverse((X_transpose)(X)))*X_transpose
		X_fold_train.insert(0,'x0',1);
	
		X_transpose = X_fold_train.T;			#X_transpose
		res = X_transpose.dot(X_fold_train);		#X_transpose.X
		res = np.linalg.inv(res);			#inverse(X_transpose.X)
		X_inverse = res.dot(X_transpose);	# X_inverse = inverse(X_transpose.X).X_transpose
	
		# Getting optimal parameter directly
		w = X_inverse.dot(y_fold_train);

		# Calculating RMSE on validation set -
		# adding x0 = 1
		X_fold_test.insert(0,'x0',1);
		rmse = rmse_cost(X_fold_test,y_fold_test,w);
		tot_err += rmse;
		print(f"For Linear regression in closed form, RMSE value on fold {c} val set is: {rmse}");
		c += 1;

	# Reporting total avg error across 5 folds
	tot_err = tot_err/5;
	print(f"For Linear regression in closed form, Average RMSE value of all folds is: {tot_err}\n");


#---------------------End of Functions definitions---------------------------------------



#---------------------------Functions calls----------------------------------------------

# Functions for corresponding parts of Question 2:

# For visualzing training data - 
feature_visualize(X_train);

# For part 1 -
part1(X_train,y_train);

# For part 2 with L1 regularization-
part2_L1(X_train,y_train);

# For part 2 with L2 regularization -
part2_L2(X_train,y_train);

# For part 3-
part3(X_train,y_train,X_test,y_test);

# For part 4 -  
part4(X_train,y_train,X_test,y_test);

# For part 5 -
part5(X_train,y_train);


#--------------------------------------end------------------------------------------------

