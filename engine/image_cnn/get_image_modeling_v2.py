# if you want know tensorflow 1.x -> tensorflow 2.x 
# tf_upgrade_v2 --infile (originfile.py) --outfile (afterfile.py)
# pip install tensorflow-cpu
# 모델 학습 후 맨 아래 쪽에서 "epoch"수 만큼 train 하고, "# Test model and check accuracy"에서 말그대로 test accuracy를 return 한다.

import tensorflow as tf
import numpy as np
import pandas as pd
import random
import glob
import os

from PIL import Image
from keras.utils import np_utils



class CNN_tensor():
    def __init__(self, image_path, label_path, model_path):
        self.image_path = image_path
        self.data = pd.read_csv(label_path, names=['hash', 'y'])
        self.model_path = model_path
        self.file_path = self.model_path + "/cnn_model"
        
    # image 가져오기
    def load_images(self):
        nb_classes = 2 # 0,1 - num class 개수
        binary_list = glob.glob(self.image_path+"/*.png")

        # 정상 프로그램 이미지 목록을 받아옴
        total_len = len(binary_list)
        BEN_TRAIN = int(round(total_len * 0.7))
        BEN_TEST = total_len - BEN_TRAIN

        # 정상 프로그램 이미지를 저장할 자료형 정의
        X_train_benign = np.empty((BEN_TRAIN, 28, 28, 1), dtype = "float32")
        y_train_benign = np.empty((BEN_TRAIN,), dtype = "uint8")
        X_test_benign = np.empty((BEN_TEST, 28, 28, 1), dtype = "float32")
        y_test_benign = np.empty((BEN_TEST,), dtype = "uint8")
        cnt = 0

        # 이미지를 불러온 후 크기를 28 x 28로 조정
        for file_name in binary_list:
            filename = file_name # test
            
            img_hash = format_spliter(os.path.split(filename)[1])
            
            im = Image.open(file_name).convert("L")
            # filename check
            label = self.data[self.data.hash==img_hash].values[0][1]
            out = im.resize((28,28)) 
        
            if cnt < BEN_TRAIN: 
                X_train_benign[cnt,:,:,0] = out
                y_train_benign[cnt,] = label # 0 is normal, 1 is malware, tag set
            else:
                X_test_benign[cnt-BEN_TRAIN,:,:,0] = out
                y_test_benign[cnt-BEN_TRAIN,] = label
                
            cnt = cnt+1
        
        y_train_benign = np.zeros(BEN_TRAIN,)
        y_test_benign = np.zeros(BEN_TEST,)

        X_train = X_train_benign.astype("float32")
        Y_train = np_utils.to_categorical(y_train_benign, nb_classes)

        X_test = X_test_benign.astype("float32")
        Y_test = np_utils.to_categorical(y_test_benign, nb_classes)

        # Last dataset
        self.x_train  = X_train
        self.x_test   = X_test
        self.y_train  = Y_train
        self.y_test   = Y_test


    # Deep CNN 실행
    def train(self):
        training_epochs = 100
        batch_size = 100
        learning_rate = 0.001
        tf.compat.v1.disable_eager_execution() # for tensorflow1 code run in tensor 2
        keep_prob = tf.compat.v1.placeholder(tf.float32)

        # Input place holders
        # placeholder : 데이터셋을 담을 computational graph
        X = tf.compat.v1.placeholder(tf.float32, [None, 28, 28 ,1]) # image 28 X 28 X 1
        Y = tf.compat.v1.placeholder(tf.float32, [None, 2])  

        # Layering
        # variable : 가중치 & 바이어스를 담을 computational graph
        W1 = tf.Variable(tf.random.normal([3, 3, 1, 32], stddev=0.01)) # 3 X 3 크기의 필터를 이용해 32개의 출력값을 얻기 위함
        L1 = tf.nn.conv2d(input=X, filters=W1, strides=[1, 1, 1, 1], padding='SAME')
        L1 = tf.nn.relu(L1)
        L1 = tf.nn.max_pool2d(input=L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        L1 = tf.nn.dropout(L1, rate=1 - (keep_prob))

        W2 = tf.Variable(tf.random.normal([3, 3, 32, 64], stddev=0.01))
        L2 = tf.nn.conv2d(input=L1, filters=W2, strides=[1, 1, 1, 1], padding='SAME')
        L2 = tf.nn.relu(L2)
        L2 = tf.nn.max_pool2d(input=L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        L2 = tf.nn.dropout(L2, rate=1 - (keep_prob))

        W3 = tf.Variable(tf.random.normal([3, 3, 64, 128], stddev=0.01))
        L3 = tf.nn.conv2d(input=L2, filters=W3, strides=[1, 1, 1, 1], padding='SAME')
        L3 = tf.nn.relu(L3)
        L3 = tf.nn.max_pool2d(input=L3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        L3 = tf.nn.dropout(L3, rate=1 - (keep_prob))
        L3_flat = tf.reshape(L3, [-1, 128 * 4 * 4])

        """
        Fully connected(FC, Dense) layer
        Final FC 128X4X4 inputs -> 625 outputs
        matmul : 행렬 곱셈하기
        """
        W4 = tf.compat.v1.get_variable("W4", shape=[128 * 4 * 4, 625], initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))
        b4 = tf.Variable(tf.random.normal([625]))
        L4 = tf.nn.relu(tf.matmul(L3_flat, W4) + b4)
        L4 = tf.nn.dropout(L4, rate=1 - (keep_prob))

        W5 = tf.compat.v1.get_variable("W5", shape=[625, 2], initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))
        b5 = tf.Variable(tf.random.normal([2]))
        logits = tf.matmul(L4, W5) + b5

        """
        Define cost/loss & optimizer
        reduce_mean : 특정 차원을 제거하고 평균을 구함 
        softmax_cross_entropy_with_logits :    
        """
        cost = tf.reduce_mean(input_tensor=tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

        # Initialize
        sess = tf.compat.v1.Session()
        sess.run(tf.compat.v1.global_variables_initializer())

        print('Learning started. It takes sometime.')
        for epoch in range(training_epochs):
            avg_cost = 0
            total_batch = int(len(self.y_train) / batch_size)

            for i in range(total_batch):

                batch_xs = self.x_train[i*batch_size:(i+1)*batch_size]
                batch_ys = self.y_train[i*batch_size:(i+1)*batch_size]

                feed_dict = {X: batch_xs, Y: batch_ys, keep_prob: 0.7}

                c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
                avg_cost += c / total_batch

            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

        print('Learning Finished!')
        
        #saver = tf.compat.v1.train.Saver()
        #saver.save(sess, self.file_path, global_step = 1000)
        #sess file save
        
        # Test model and check accuracy
        correct_prediction = tf.equal(tf.argmax(input=logits, axis=1), tf.argmax(input=Y, axis=1))
        accuracy = tf.reduce_mean(input_tensor=tf.cast(correct_prediction, tf.float32))
        acc = sess.run(accuracy, feed_dict={X: self.x_test, Y: self.y_test, keep_prob: 1})
        print("acc : ", acc)
        
    def predict_do(self):
        new_sess = tf.compat.v1.Session()
        saver = tf.compat.v1.train.import_meta_graph(self.file_path+'-1000.meta')
        saver.restore(new_sess,tf.train.latest_checkpoint(self.model_path))
        #load model

        # Test model and check accuracy
        correct_prediction = tf.equal(tf.argmax(input=logits, axis=1), tf.argmax(input=Y, axis=1))
        accuracy = tf.reduce_mean(input_tensor=tf.cast(correct_prediction, tf.float32))
        acc = sess.run(accuracy, feed_dict={X: self.x_test, Y: self.y_test, keep_prob: 1})

        return acc


def main():
    path = "./sample/target_images"
    label_path = ""
    cn = CNN_tensor(path, label_path)
    cn.load_images()
    cn.train()

if __name__ == '__main__':
    main()