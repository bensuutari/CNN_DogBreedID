import numpy as np
import tensorflow as tf
import random
import os
from os import listdir
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
import pickle
import pandas as pd
from skimage.color import rgb2gray
from random import shuffle
import time

num_epochs=20

imsize=52
train_size=9600#the amount of the data to allocate to the CNN training

filepath=os.getcwd()+'/data/train/'
trainlabels=pd.read_csv(os.getcwd()+'/data/labels.csv')

allbreeds=list(set(trainlabels['breed']))
n_classes=len(allbreeds)

#create indices for all images and shuffle them
shuffleindices=range(0,len(trainlabels))
shuffle(shuffleindices)
trainlabels_id=np.asarray(trainlabels['id'])[shuffleindices]
trainlabels_breed=np.asarray(trainlabels['breed'])[shuffleindices]




#make a function prepares data to be fed into CNN by resizing, normalizing and reshaping by batch size
def load_data(files):
	imagebatch=np.zeros((len(files),imsize*imsize))
	counter=0
	imageloadcounter=1
	for i in files:
		im=Image.open(filepath+i+'.jpg').convert('L')
		im=im.resize((imsize,imsize),Image.NEAREST)
		im=np.asarray(im)/255.0
		imagebatch[counter,]=im.reshape((imsize*imsize,))
		counter+=1
		if counter%1000==0:
			print str(imageloadcounter*1000)+' images loaded'
			imageloadcounter+=1
	return imagebatch



#make a plot function so we can visualize how our resized images in a batch look
def showdogbatch(images):

    for i in range(0,images.shape[0]):
    	plt.imshow(images[i,].reshape((100,100)),cmap='gray')
    	plt.show()

def label2onehot(data):
	onehotarray=np.zeros((len(data),len(allbreeds)))
	for i in range(0,len(data)):
		onehotarray[i,allbreeds.index(data[i])]=1
	return np.asarray(onehotarray)


#The convolutional neural network constructor
x = tf.placeholder('float', [None, imsize*imsize])
y = tf.placeholder('float',[None,n_classes])

keep_rate = 1.0#0.8
#keep_prob = tf.placeholder(tf.float32)

doglabels=label2onehot(trainlabels_breed[0:train_size])
doglabelstest=label2onehot(trainlabels_breed[train_size:])

trainbatch=load_data(trainlabels_id[0:train_size])
testbatch=load_data(trainlabels_id[train_size:])#allocate any data not in the training group to testing group



def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxpool2d(x):
    #                        size of window         movement of window
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')



def convolutional_neural_network(x):
    weights = {'W_conv1':tf.Variable(tf.random_normal([5,5,1,32])),
               'W_conv2':tf.Variable(tf.random_normal([5,5,32,64])),
               'W_fc':tf.Variable(tf.random_normal([(imsize/4)*(imsize/4)*64,1024])),#imsize/4 because each max pool has a stride of 2, so it reduces dimensionality by 2 twice or 1/2^2
               'out':tf.Variable(tf.random_normal([1024, n_classes]))}

    biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
               'b_conv2':tf.Variable(tf.random_normal([64])),
               'b_fc':tf.Variable(tf.random_normal([1024])),
               'out':tf.Variable(tf.random_normal([n_classes]))}

    x = tf.reshape(x, shape=[-1, imsize, imsize, 1])

    conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool2d(conv1)
    
    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool2d(conv2)

    fc = tf.reshape(conv2,[-1, (imsize/4)*(imsize/4)*64])#imsize/4 because each max pool has a stride of 2, so it reduces dimensionality by 2 twice or 1/2^2
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate)

    output = tf.matmul(fc, weights['out'])+biases['out']

    return output

def train_neural_network(x):
    prediction = convolutional_neural_network(x)
    print 'prediction shape='
    print(prediction.get_shape())
    print 'logits shape='
    print(y.get_shape())
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))

    optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)
    
    with tf.Session() as sess:
        #sess.run(tf.initialize_all_variables())
        sess.run(tf.global_variables_initializer())
        
        
        acclist=list()
        losslist=list()
        #create a tf saver to save models
        saver=tf.train.Saver()
        for epoch in range(num_epochs):
            startepoch=time.time()
            epoch_loss = 0
            _, c = sess.run([optimizer, cost], feed_dict={x: trainbatch, y: doglabels})
            epoch_loss += c
            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            acclist.append(accuracy.eval({x:testbatch, y:doglabelstest}))
            losslist.append(epoch_loss)
            print('Epoch '+ str(epoch+1) + '/'+str(num_epochs)+','+' loss:'+str(epoch_loss),'Test Accuracy:',accuracy.eval({x:testbatch, y:doglabelstest}))
            endepoch=time.time()
            print 'Epoch Took: '+str((endepoch-startepoch)/60.0)+' minutes'
            saver.save(sess, os.getcwd()+'/models/dog_breed_id_CNN_model')
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:testbatch, y:doglabelstest}))
        return acclist,losslist

acclist,losslist=train_neural_network(x)




#showdogbatch(trainbatch)




