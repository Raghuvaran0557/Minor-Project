import os
import cv2
import numpy as np
from sklearn_extra.cluster import KMedoids
import matplotlib.pyplot as plt

# Non-Binary Image Classification using Convolution Neural Networks
'''
path = 'shapes'

labels = []
X_train = []
Y_train = []

def getID(name):
    index = 0
    for i in range(len(labels)):
        if labels[i] == name:
            index = i
            break
    return index        
    

for root, dirs, directory in os.walk(path):
    for j in range(len(directory)):
        name = os.path.basename(root)
        if name not in labels:
            labels.append(name)
print(labels)

for root, dirs, directory in os.walk(path):
    for j in range(len(directory)):
        name = os.path.basename(root)
        print(name+" "+root+"/"+directory[j])
        if 'Thumbs.db' not in directory[j]:
            img = cv2.imread(root+"/"+directory[j],0)
            img = cv2.resize(img, (64,64))
            X_train.append(img.ravel())
            Y_train.append(getID(name))
        
X_train = np.asarray(X_train)
Y_train = np.asarray(Y_train)
print(Y_train)
np.save('model/X.txt',X_train)
np.save('model/Y.txt',Y_train)

'''
X_train = np.load('model/X.txt.npy')
Y_train = np.load('model/Y.txt.npy')

X_train = X_train.astype('float32')
X_train = X_train/255
    
test = X_train[3]
test = test.reshape(64,64)
cv2.imshow("aa",test)
cv2.waitKey(0)
indices = np.arange(X_train.shape[0])
np.random.shuffle(indices)
X_train = X_train[indices]
Y_train = Y_train[indices]

kmedoids = KMedoids(n_clusters=4, metric='euclidean',random_state=0,max_iter=10000)
kmedoids.fit(X_train)
predict = kmedoids.labels_

images = []
for i in range(len(predict)):
    if len(images) < 20:
        if predict[i] == 0:
            images.append(X_train[i])

images1 = []
for i in range(len(predict)):
    if len(images1) < 20:
        if predict[i] == 1:
            images1.append(X_train[i])

images2 = []
for i in range(len(predict)):
    if len(images2) < 20:
        if predict[i] == 2:
            images2.append(X_train[i])

images3 = []
for i in range(len(predict)):
    if len(images3) < 20:
        if predict[i] == 3:
            images3.append(X_train[i])            

def plotShapes(array, title):
    w = 64
    h = 64
    fig = plt.figure(figsize=(8, 8))
    columns = 4
    rows = 5
    index = 0
    for i in range(1, columns*rows +1):
        img = array[index].reshape(64,64)
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
        index += 1
    plt.title(title)    
    plt.show()

plotShapes(images,"Shapes in Cluster 1")
plotShapes(images1,"Shapes in Cluster 2")
plotShapes(images2,"Shapes in Cluster 3")
plotShapes(images3,"Shapes in Cluster 4")
