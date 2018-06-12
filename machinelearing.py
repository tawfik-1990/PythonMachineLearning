import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import svm
from sklearn.cluster import KMeans , MeanShift 
from sklearn.decomposition import PCA
from matplotlib import style 
style.use ("ggplot")




# Supervised learning with Classification - SVC

my_input = np.array([[3,2],
	[6,6],
	[2.6,3],
	[7,8],
	[3.5,5],
	[6,11]])



my_output = [0,1,0,1,0,1]


my_model=svm.SVC(kernel= 'linear',C=1.0)
my_model.fit(my_input,my_output)


print("SVC predict [0.5,0.8] :" , my_model.predict([[0.5,0.8] ]))

print("SVC predict [8.5,10] :" , my_model.predict([[8.5,10] ]))


plt.scatter(my_input[:,0], my_input[:,1], c= my_output)
plt.scatter(0.5,0.8,c ='r')
plt.scatter(8.5,10,c ='r')

plt.show()





# Supervised learning with Regression - SVR


X = np.array([[3,2],
	[6,6],
	[2.6,3],
	[7,8],
	[3.5,5],
	[6,11]])



Y = [0,1,0,1,0,1]


my_clf=svm.SVR(kernel= 'linear',C=1.0)
my_clf.fit(X,Y)


print("SVR predict [0.5,0.8] :" , my_clf.predict([[0.5,0.8] ]))

print("SVR predict [8.5,10] :" , my_clf.predict([[8.5,10] ]))


plt.scatter(my_input[:,0], my_input[:,1], c= Y)
plt.scatter(0.5,0.8,c ='r')
plt.scatter(8.5,10,c ='r')

plt.show()







# Unsupervised learning with Flat Clustering - K-Means

my_input1 = np.array([[3,2],
	[6,6],
	[2.6,3],
	[7,8],
	[3.5,5],
	[6,11]])


my_model1=KMeans(n_clusters=3)

my_model1.fit(my_input1)


print("cluster_centers :\n" , my_model1.cluster_centers_)


print("labels" , my_model1.labels_)

colors=["g.","r.","c.","y."]
plt.scatter(my_input1[:,0], my_input1[:,1], c= my_model1.labels_)
plt.scatter(my_model1.cluster_centers_[:,0],my_model1.cluster_centers_[:,1], marker = "x" ,s =250 ,linewidths=5)


plt.show()





# Unsupervised learning with Hiearchical Clustering - MeanShift

my_input2 = np.array([[4,3],[6.5,7.5],[2.6,4],[7,8],[3.5,5],[6,9],
	                  [3,5],[6,8],[3.5,3],[6.5,8],[4,4],[3,3],
	                  [3,4],[3.5,4],[3.4,3.7],[3.3,4.5],[3.8,3]])

	
	
	



my_model2=MeanShift()

my_model2.fit(my_input2)


print("cluster_centers :\n" , my_model2.cluster_centers_)


print("labels" , my_model2.labels_)

colors=["g.","r.","c.","y."]
plt.scatter(my_input2[:,0], my_input2[:,1], c= my_model2.labels_)
plt.scatter(my_model2.cluster_centers_[:,0],my_model2.cluster_centers_[:,1], marker = "x" ,s =250 ,linewidths=5)


plt.show()







# Unsupervised learning with Dimensinality reduction - PCA

rng = np.random.RandomState(1)
my_input3= np.dot(rng.rand(2,2),rng.randn(2,100)).T	
	
	



my_model3=PCA(n_components=1)
my_model3.fit(my_input3)
my_input3_pca=my_model3.transform(my_input3)




print("the origanal shape" ,my_input3.shape)


print("the transformed shape" , my_input3_pca.shape)

my_input3_newvalues=my_model3.inverse_transform(my_input3_pca)
plt.scatter(my_input3[:,0], my_input3[:,1])
plt.scatter(my_input3_newvalues[:,0],my_input3_newvalues[:,1])


plt.show()


