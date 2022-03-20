# Name - Sagar Suman
# Roll No. 2019197
# ML Assignment - 1

# Question 1 - part 2


#Importing libraries
import numpy as np
import idx2numpy
import matplotlib.pyplot as plt
import random
from sklearn.manifold import TSNE


# A) - Loading the dataset using idx2numpy 

# X_train - 60000 samples of (28 X 28)
X_train = idx2numpy.convert_from_file('train-images.idx3-ubyte');
#print(X_train.shape);

# labels for X_train - 60000 
y_train = idx2numpy.convert_from_file('train-labels.idx1-ubyte');
#print(y_train.shape);

#  X_test - 10000 samples of (28 X 28)
X_test = idx2numpy.convert_from_file('t10k-images.idx3-ubyte');
#print(X_test.shape);

# labels for X_test - 10000
y_test = idx2numpy.convert_from_file('t10k-labels.idx1-ubyte');
#print(y_test.shape);

#------------------------------------------------------

# B) - Visualizing two random images from dataset 


#Defining function for this part -
def Q1p2_visualize(X_train):
	
	# Creating two random number in range 0 to 59999-
	pic1 = random.randint(0,59999);
	pic2 = random.randint(0,59999);
	#print(pic1," ",pic2);

	plt.figure('Random image 1:');
	plt.imshow(X_train[pic1]); # using imshow on first image

	plt.figure('Random image 2:');
	plt.imshow(X_train[pic2]); # using imshow on second image
	#Showing images-
	plt.show();


# Calling above function-
Q1p2_visualize(X_train);


#------------------------------------------------------

# C) -

# Step 1 - Taking out 1000 samples from each of 10 class
c = 0;
while(c < 10):
	# Taking out all labels of particular digit
	temp = np.where(y_train == c); 
	# Coverting the above in numpy array
	temp = np.asarray(temp);
	# Flattening-
	temp = temp.flatten();
	# Taking out 1000 random samples of each digit
	temp = np.random.choice(temp,1000,replace = False);
	if(c == 0):
		index = temp;
	else:
		# Storing samples indexes-
		index = np.append(index,temp);	
	c += 1;
# New data set of 10,000 samples in which each class have 1000 samples
X_new_train = X_train[index];
y_new_train = y_train[index];
# Reshaping -
X_new_train = X_new_train.reshape(10000,784);

# Step 3 - Using TSNE for dimensionality reduction to 2
tsne = TSNE(n_components = 2);
X_train_reduced = tsne.fit_transform(X_new_train);

# Step 4 - plotting 
# Taking out x and y coordinate
x = X_train_reduced[:,0];
y = X_train_reduced[:,1];
# colors - 
plt.figure('tSNE result of data reduced to 2 dimension'); 
color = ['black','violet','lightgreen','grey','blue','cyan','green','yellow','orange','red'];
count = 0;
r = 1000;
# plotting scatter plot and assigning respective labels
while(count < 10):
	plt.scatter(x[r-1000:r-1],y[r-1000:r-1],c=color[count],label=count);
	count += 1;
	r += 1000;


plt.title('MNIST DATASET with 2 dimensions');
plt.legend(bbox_to_anchor=(1,1));
plt.tight_layout();
# showing
plt.show();
