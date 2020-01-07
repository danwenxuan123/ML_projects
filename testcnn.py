# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 17:49:29 2017

@author: Dan Wenxuan
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 18:06:19 2017

@author: Dan Wenxuan
"""
import cv2
import os
import tensorflow as tf
import numpy as np
#load image
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

#randomly shuffle images    
def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b

#define result    
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
  
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
    
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
  
def main():
    
    Young=load_images_from_folder('C:/Users/Dan Wenxuan/Desktop/ML Project/1~30')
    Adult=load_images_from_folder('C:/Users/Dan Wenxuan/Desktop/ML Project/30~60')
    Old=load_images_from_folder('C:/Users/Dan Wenxuan/Desktop/ML Project/60+')
    
    NumYoung=len(Young)
    NumAdult=len(Adult)
    NumOld=len(Old)

    YoungArr=np.asarray(Young)
    AdultArr=np.asarray(Adult)
    OldArr=np.asarray(Old)
    

    
    AugYoung=np.zeros((NumYoung,64,64))
    AugAdult=np.zeros((NumAdult,64,64))
    AugOld=np.zeros((NumOld,64,64))
    for i in range(NumYoung):
        im=cv2.resize(YoungArr[i],(64, 64)) 
        AugYoung[i]=rgb2gray(im)
           
    for i in range(NumAdult):
        im=cv2.resize(AdultArr[i],(64, 64)) 
        AugAdult[i]=rgb2gray(im)
    
    for i in range(NumOld):
        im=cv2.resize(OldArr[i],(64, 64)) 
        AugOld[i]=rgb2gray(im)
        
    TYoung=load_images_from_folder('C:/Users/Dan Wenxuan/Desktop/ML Project/T1~30')
    TAdult=load_images_from_folder('C:/Users/Dan Wenxuan/Desktop/ML Project/T30~60')
    TOld=load_images_from_folder('C:/Users/Dan Wenxuan/Desktop/ML Project/T60+')
    
    TNumYoung=len(TYoung)
    TNumAdult=len(TAdult)
    TNumOld=len(TOld)
    print(TNumYoung)
    print(TNumAdult)
    print(TNumOld)
    print(NumYoung)
    print(NumAdult)
    print(NumOld)
    TYoungArr=np.asarray(TYoung)
    TAdultArr=np.asarray(TAdult)
    TOldArr=np.asarray(TOld)

    

    TAugYoung=np.zeros((TNumYoung,64,64))
    TAugAdult=np.zeros((TNumAdult,64,64))
    TAugOld=np.zeros((TNumOld,64,64))
    for i in range(TNumYoung):
        im=cv2.resize(TYoungArr[i],(64, 64)) 
        TAugYoung[i]=rgb2gray(im)
           
    for i in range(TNumAdult):
        im=cv2.resize(TAdultArr[i],(64, 64)) 
        TAugAdult[i]=rgb2gray(im)
    
    for i in range(TNumOld):
        im=cv2.resize(TOldArr[i],(64, 64)) 
        TAugOld[i]=rgb2gray(im)

    
    ValSet=np.vstack((AugYoung,AugAdult,AugOld))
    YoungLabels=np.array([[1,0,0],]*NumYoung)
    AdultLabels=np.array([[0,1,0],]*NumAdult)
    OldLabels=np.array([[0,0,1],]*NumOld)
    ValLabels=np.concatenate((AdultLabels,OldLabels,YoungLabels),axis=0)
    TestSet=np.vstack((TAugYoung,TAugAdult,TAugOld))
    TYoungLabels=np.array([[1,0,0],]*TNumYoung)
    TAdultLabels=np.array([[0,1,0],]*TNumAdult)
    TOldLabels=np.array([[0,0,1],]*TNumOld)
    TestLabels=np.concatenate((TAdultLabels,TOldLabels,TYoungLabels),axis=0)
    [TestSet,TestLabels]=shuffle_in_unison(TestSet,TestLabels)
    [ValSet,ValLabels]=shuffle_in_unison(ValSet,ValLabels)
    TestSet=TestSet.reshape(TNumYoung+TNumAdult+TNumOld,64*64)
    ValSet=ValSet.reshape(10000,64*64)
    """
    TrainingSmile=TrainingSmile.reshape(2*numtotal,64*64)
    TrainingLabels=np.column_stack((TrainingLabels,1-TrainingLabels))
    ValSet=ValSet.reshape(TNumSmile+TNumNotSmile,64*64)
    ValLabels=np.column_stack((ValLabels,1-ValLabels))
    """
    
    x = tf.placeholder(tf.float32, shape=[None, 64*64])
    y_ = tf.placeholder(tf.float32, shape=[None, 3])
    
   
    #first convolution layer
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
   
    x_image=tf.reshape(x, [-1, 64, 64, 1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
   
    #first max pool layer
    h_pool1 = max_pool_2x2(h_conv1)
   
    #second convolution layer 
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
   
    #second max pool layer
    h_pool2 = max_pool_2x2(h_conv2)
   
    #full connected layer 
    W_fc1 = weight_variable([16 * 16 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 16*16*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    
    #dropout
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
    #readout layer
    W_fc2 = weight_variable([1024, 3])
    b_fc2 = bias_variable([3])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(400):
            for j in range(100):
                batch_x=ValSet[j*100:(j+1)*100,:]
                batch_y=ValLabels[j*100:(j+1)*100,:]
                train_accuracy = accuracy.eval(feed_dict={x:batch_x, y_:batch_y, keep_prob: 1.0})
                print('step %d, training accuracy %g' % (i, train_accuracy))
            train_step.run(feed_dict={x:batch_x, y_: batch_y, keep_prob: 0.5})    
            
        print('test accuracy %g' % accuracy.eval(feed_dict={x: TestSet, y_: TestLabels, keep_prob: 1.0}))
        save_path = saver.save(sess, "C:/Users/Dan Wenxuan/Desktop/ML Project/model.ckpt")
        print("Model saved in file: %s" % save_path)
if __name__ == "__main__": main()

