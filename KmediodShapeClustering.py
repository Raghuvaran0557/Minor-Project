from tkinter import *
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import pandas as pd
import numpy as np
import os

import cv2
import numpy as np
from sklearn_extra.cluster import KMedoids


main = tkinter.Tk()
main.title("Application and evaluation of a K-Medoids-based shape clustering method for an articulated design space")
main.geometry("1200x1200")

global dataset
global X, Y, clusters, shape1, shape2, shape3, shape4

   
def uploadDataset():
    global dataset
    text.delete('1.0', END)
    filename = filedialog.askdirectory(initialdir=".")
    text.insert(END,str(filename)+" Dataset Loaded\n\n")
    pathlabel.config(text=str(filename)+" Dataset Loaded\n\n")
    

def preprocessDataset():
    text.delete('1.0', END)
    global dataset
    global X, Y
    if os.path.exists("model/X.txt.npy"):
        X = np.load('model/X.txt.npy')
        Y = np.load('model/Y.txt.npy')
    else:
        X = []
        Y = []
        for root, dirs, directory in os.walk(dataset):
            for j in range(len(directory)):
                name = os.path.basename(root)
                print(name+" "+root+"/"+directory[j])
                if 'Thumbs.db' not in directory[j]:
                    img = cv2.imread(root+"/"+directory[j],0)
                    img = cv2.resize(img, (64,64))
                    X.append(img.ravel())
                    Y.append(getID(name))
        X = np.asarray(X)
        Y = np.asarray(Y)
        np.save('model/X.txt',X)
        np.save('model/Y.txt',Y)
    X = X.astype('float32')
    X = X/255
    test = X[3]
    test = test.reshape(64,64)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    text.insert(END,"Total images found in dataset : "+str(X.shape[0])+"\n\n")
    text.insert(END,"Shapes found in dataset are : Star, Circle, Triangle and Rectangle\n\n")
    text.update_idletasks()
    test = cv2.resize(test,(300,300))
    cv2.imshow("Sample Process Image",test)
    cv2.waitKey(0)

def runClustering():
    global X, Y, clusters
    global shape1, shape2, shape3, shape4
    text.delete('1.0', END)
    kmedoids = KMedoids(n_clusters=4, metric='euclidean',random_state=0,max_iter=10000)
    kmedoids.fit(X)
    clusters = kmedoids.labels_
    text.insert(END,"Clusters List = "+str(clusters.tolist())+"\n\n")
    count1 = 0
    count2 = 0
    count3 = 0
    count4 = 0
    shape1 = []
    for i in range(len(clusters)):
        if clusters[i] == 0:
            count1 += 1
        if len(shape1) < 20:
            if clusters[i] == 0:
                shape1.append(X[i])

    shape2 = []
    for i in range(len(clusters)):
        if clusters[i] == 1:
            count2 += 1
        if len(shape2) < 20:
            if clusters[i] == 1:
                shape2.append(X[i])

    shape3 = []
    for i in range(len(clusters)):
        if clusters[i] == 2:
            count3 += 1
        if len(shape3) < 20:
            if clusters[i] == 2:
                shape3.append(X[i])

    shape4 = []
    for i in range(len(clusters)):
        if clusters[i] == 3:
            count4 += 1
        if len(shape4) < 20:
            if clusters[i] == 3:
                shape4.append(X[i])
    text.insert(END,"Clustering process completed\n\n")
    text.insert(END,"Total shapes found in Cluster1 : "+str(count1)+"\n\n")
    text.insert(END,"Total shapes found in Cluster2 : "+str(count2)+"\n\n")
    text.insert(END,"Total shapes found in Cluster3 : "+str(count3)+"\n\n")
    text.insert(END,"Total shapes found in Cluster4 : "+str(count4)+"\n\n")

def plotShapes(array, titles):
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
    fig.suptitle(titles, fontsize=20)    
    

def visualization():
    global shape1, shape2, shape3, shape4
    plotShapes(shape1,"Shapes in Cluster 1")
    plotShapes(shape2,"Shapes in Cluster 2")
    plotShapes(shape3,"Shapes in Cluster 3")
    plotShapes(shape4,"Shapes in Cluster 4")
    plt.show()

def close():
    main.destroy()
    
font = ('times', 14, 'bold')
title = Label(main, text='Application and evaluation of a K-Medoids-based shape clustering method for an articulated design space')
title.config(bg='DarkGoldenrod1', fg='black')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=5,y=5)

font1 = ('times', 12, 'bold')
uploadButton = Button(main, text="Upload Shapes Dataset", command=uploadDataset)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='brown', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=560,y=100)

preprocessButton = Button(main, text="Preprocess & Hamming Distance Calculation", command=preprocessDataset)
preprocessButton.place(x=50,y=150)
preprocessButton.config(font=font1)

hybridMLButton = Button(main, text="Run K-Medoids Clustering Algorithm", command=runClustering)
hybridMLButton.place(x=50,y=200)
hybridMLButton.config(font=font1)

snButton = Button(main, text="Similar Shapes Visualization from Clusters", command=visualization)
snButton.place(x=50,y=250)
snButton.config(font=font1)

snButton = Button(main, text="Exit", command=close)
snButton.place(x=50,y=300)
snButton.config(font=font1)


font1 = ('times', 12, 'bold')
text=Text(main,height=25,width=100)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=400,y=150)
text.config(font=font1)


main.config(bg='LightSteelBlue1')
main.mainloop()
