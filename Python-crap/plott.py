import matplotlib.pyplot as plt;
import numpy as np;

x = [1,2,3,4,5,6];
y1 = [1,2,3,4,5,6];
y2 = [1,4,9,16,25,36];

plt.scatter(x,y1,label = 'y = x');
plt.scatter(x,y2,label='y = x^2');

plt.legend(bbox_to_anchor=(1,1));
plt.tight_layout();
#plt.show();

x = np.asarray([[[1,2],[4,5]], [[7,8],[9,10]]]);
print(x);
print("x.shape",x.shape);
y = x.reshape(2,4);
print("y",y);
print("y.shape",y.shape);

