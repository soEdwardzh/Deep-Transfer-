import tensorflow as tf 
import matplotlib.pyplot as plt
import os
os.environ['CUDA_VISIBLE_DEVICES']='3'
import numpy as np
from  Class_RevNet import RevNet
import math

def read_and_decode(filename,Mirror=True):
    
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                           'domain_label':tf.FixedLenFeature([], tf.int64)
                                       }) 
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [227, 227, 3])
    img = tf.cast(img,dtype=tf.float32)
    if Mirror:
        img = tf.image.random_flip_left_right(img)
#        img = tf.image.random_hue(img,max_delta=0.2)
#        img = tf.image.random_contrast(img,lower=0.5,upper=1.0)
    domain_label = tf.cast(features['domain_label'], tf.int32)
    label = tf.cast(features['label'], tf.int32)

    return img, label, domain_label

#Amazon,2700,Webcam,800,DSLR:500
img_amazon,label_amazon,domain_label_amazon=read_and_decode('/home/zhengh/Tensorflow/DAN_TF/Data/Amazon_with_DomainLabel_Source.tfrecords')
img_webcam,label_webcam,domain_label_webcam=read_and_decode('/home/zhengh/Tensorflow/DAN_TF/Data/Webcam_with_DomainLabel_Target.tfrecords')
test_amazon,test_amazon_label,test_domain_amazon=read_and_decode('/home/zhengh/Tensorflow/DAN_TF/Data/Amazon_with_DomainLabel_Source.tfrecords',Mirror=False)
test_webcam,test_webcam_label,test_domain_webcam=read_and_decode('/home/zhengh/Tensorflow/DAN_TF/Data/Webcam_with_DomainLabel_Target.tfrecords',Mirror=False)
batch_source,labels_source,domain_labels_source = tf.train.shuffle_batch([img_amazon, label_amazon,domain_label_amazon], 
                                                                         batch_size=32,capacity=500, min_after_dequeue=200)

batch_target,labels_target,domain_labels_target = tf.train.shuffle_batch([img_webcam, label_webcam,domain_label_webcam], 
                                                                         batch_size=32,capacity=500, min_after_dequeue=200)
Test_Target,Testlabels_Target,Test_Target_DL = tf.train.batch([test_webcam,test_webcam_label,test_domain_webcam],
                                                batch_size=10, capacity=300) 
Test_Source,Testlabels_Source,Test_Source_DL = tf.train.batch([test_amazon,test_amazon_label,test_domain_amazon],
                                                batch_size=10, capacity=300)

TrainBatch = tf.concat(0,[batch_source,batch_target])
Labels = tf.concat(0,[labels_source,labels_target])
Domain_Labels = tf.concat(0,[domain_labels_source,domain_labels_target])




Global_Step = tf.Variable(0.0,trainable=False)
Init = np.load('/home/zhengh/Tensorflow/DAN_TF/Model/Alexnet.npy').item()
Net = RevNet(pretrain=True,Initializer=Init)
domain_tradeoff = tf.placeholder(dtype=tf.float32,name='domain_tradeoff')
condition_entropy_tradeoff = tf.placeholder(dtype=tf.float32,name='ce_tradeoff')
SDA = tf.placeholder(dtype=tf.float32,name='SDA_Tradeoff')

with tf.variable_scope('train') as scope:
    Trn_OP,Domain_Loss,Label_Loss,CE_Loss,SDA_loss = Net.Train(TrainBatch,Labels,Domain_Labels,
                                           Domain_Tradeoff=domain_tradeoff,
                                           base_lr=0.001,Global_step=Global_Step,
                                           condition_entropy_rate=condition_entropy_tradeoff,SDA=SDA)
    scope.reuse_variables() 
    Accuracy_Source,DomainAcc_S = Net.Accuracy(Test_Source,Testlabels_Source,Test_Source_DL)                                          
    Accuracy_Target,DomainAcc_T = Net.Accuracy(Test_Target,Testlabels_Target,Test_Target_DL)
    

init = tf.initialize_all_variables()

config=tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.7
#config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config = config)  
sess.run(init)
threads = tf.train.start_queue_runners(sess=sess)

def Domain_Tradeoff(global_step):
    if sess.run(global_step)<=10000:
        result = (2/(1+math.e**(-10*sess.run(global_step)/(10000.0))))-1
    
    elif sess.run(global_step)>10000:
        result = (2/(1+math.e**(-10)))-1
#    result= ((2/(1+math.e**(sess.run(global_step)/(-2000.0))))-1)*0.1
    return result * 0.0

def Conditon_Tradeoff(global_step):
    #result = ((2/(1+math.e**(sess.run(global_step)/(-10000.0))))-1)*0.0
    return 0.0
def SDA_Tradeoff(global_step):
#    result= ((2/(1+math.e**(sess.run(global_step)/(-2000.0))))-1)*1.0
#    if sess.run(global_step)>=2000:
#        result = 0.0
#    else:
#        result = 0.0
    return 0.0
    
test_iter = 80
for i in range(30001):
    sess.run(Trn_OP,feed_dict={domain_tradeoff:Domain_Tradeoff(Global_Step),
                                        condition_entropy_tradeoff:Conditon_Tradeoff(Global_Step),
                                        SDA:SDA_Tradeoff(Global_Step)})
    if i%100==0 and i!=0:
         acc_ds=0
         acc_dt=0
         acc_s=0
         acc_t=0
         d_loss=0
         l_loss=0
         c_loss=0
         sda_loss = 0
         for k in range(test_iter):
             tem_ds,tem_dt,tem_acc_t,tem_acc_s,dl,ll,cl,sl = sess.run([DomainAcc_S,DomainAcc_T,
                                                      Accuracy_Target,Accuracy_Source,
                                                      Domain_Loss,Label_Loss,CE_Loss,SDA_loss],
                                      feed_dict={domain_tradeoff:Domain_Tradeoff(Global_Step),
                                      condition_entropy_tradeoff:Conditon_Tradeoff(Global_Step),SDA:SDA_Tradeoff(Global_Step)})
             acc_t += tem_acc_t
             acc_s += tem_acc_s
             acc_ds += tem_ds
             acc_dt += tem_dt
             d_loss += dl
             l_loss += ll
             c_loss += cl
             sda_loss += sl
         acc_t = acc_t/test_iter
         acc_s = acc_s/test_iter
         acc_D = (acc_ds+acc_dt)/(test_iter*2)
         d_loss = d_loss/test_iter
         l_loss = l_loss/test_iter
         c_loss = c_loss/test_iter
         sda_loss = sda_loss/test_iter
         print 'Target Accuracy is',acc_t,'\nSource Accuracy is',acc_s,'\nDomain Accuracy is',\
         acc_D,'\nDomain Loss is',d_loss,'\nLabel Loss is',l_loss,'\nConditon_EntropyLoss is',c_loss,\
         '\nSDA Loss is',sda_loss
         print'========================This is',sess.run(Global_Step),'Test Times============================='
#         
#a,b=sess.run([img_webcam,label_webcam])
#plt.imshow(np.uint8(a))
