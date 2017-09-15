
import tensorflow as tf

Alex_MEAN = [103.939, 116.779, 123.68]

class RevNet():
    def __init__(self,pretrain,Initializer):
        self.pretrain = pretrain
        if self.pretrain:
            self.initializer = Initializer
        else:
            self.initializer = None
        return
    
    def _get_conv_filter(self,name,shape,dtype=tf.float32,initializer=None):#the shape(height,weight,in,output)
               
        if shape!=self.initializer[name]['weights'].shape:
            print 'The params',shape,'and',self.initializer[name]['weights'].shape,'MisMatch!!'
            exit()
        else:
            print 'Shape=',shape,'match:',self.initializer[name]['weights'].shape,'!!'
        if self.pretrain:
            weight = tf.get_variable(name=name+'_weights',initializer=self.initializer[name]['weights'])
        else:
            weight = tf.get_variable(name=name+'_weight',shape=shape,
                        initializer=tf.truncated_normal_initializer(stddev=initializer, dtype=dtype),
                            dtype=dtype)
            

        return weight

    def _get_bias(self, name,shape,dtype=tf.float32,initializer=None,pretrain_close=False):
        
        
        if self.pretrain and pretrain_close==False:
            
            bias = tf.get_variable(name = name+'_biases',initializer=self.initializer[name]['biases'])
        else:
            bias = tf.get_variable(name=name+'_biases',shape=shape,
                        initializer=tf.constant_initializer(value=initializer, dtype=dtype),
                            dtype=dtype)

        return bias

    def _get_fc_weight(self, name,shape,dtype=tf.float32,initializer=None,pretrain_close=False):
        if pretrain_close == False:
            if shape!=self.initializer[name]['weights'].shape :
                print 'The params',shape, 'and',self.initializer[name]['weights'].shape,'MisMatch!!'
                exit()
            else:
                print 'Shape=',shape,'match:',self.initializer[name]['weights'].shape,'!!'
        if self.pretrain and  pretrain_close==False:
            weight = tf.get_variable(name = name+'_weights',initializer=self.initializer[name]['weights'])
        else:
            weight = tf.get_variable(name=name+'_weights',shape=shape,
                        initializer=tf.truncated_normal_initializer(stddev=initializer, dtype=dtype),
                            dtype=dtype)
        return weight
        
    def _max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],padding='VALID', name=name)    
        
        
    def _conv_layer(self,bottom,shape,name,pad,stride,dtype=tf.float32,initializer=(0.01,0),
                    group=1,wd=0.0005,bd=0.0):
        with tf.variable_scope(name) :
            weight = self._get_conv_filter(name,shape,dtype,initializer[0])
            conv_biases = self._get_bias(name,shape[-1],dtype,initializer[1])            
            if group == 1:
                conv = tf.nn.conv2d(bottom, weight, stride, padding=pad)
                bias = tf.nn.bias_add(conv, conv_biases)
            else:
                conv_groups = tf.split(3, group, bottom)
                weights_groups = tf.split(3, group, weight)
                conv = [tf.nn.conv2d(i,k, stride, padding=pad) for i, k in zip(conv_groups,weights_groups)]
                conv = tf.concat(3, conv)
                bias = tf.nn.bias_add(conv, conv_biases)
        
        if self.Phase == 'Train':
            self._variable_decay(weight,wd)
            self._variable_decay(conv_biases,bd)
            

        return bias        
        
    def _RELU(self,bottom):
        relu = tf.nn.relu(bottom)
        return relu        
    
    
    def _LRN(self,bottom,name):
        norm = tf.nn.lrn(bottom,depth_radius=2,alpha=2e-05,beta=0.75,name=name)
        return norm        
    
    
    def _reshape_conv_to_fc(self,bottom):
        shape = bottom.get_shape().as_list()
        dim = 1
        for d in shape[1:]:
                 dim *= d
        x = tf.reshape(bottom, [-1, dim])
        return x
        
    def _fc_layer(self, bottom,output_num,name,
                 dtype=tf.float32,initializer=None,option=False,wd=0.0005,bd=0.0):
        
        dim = bottom.get_shape().as_list()[1]
        shape = (dim,output_num)
        
        with tf.variable_scope(name):
            
            weights = self._get_fc_weight(name,shape,dtype,initializer[0],pretrain_close=option)
            biases = self._get_bias(name,output_num,dtype,initializer[1],pretrain_close=option)
            fc = tf.nn.bias_add(tf.matmul(bottom,weights),biases)
            
        if self.Phase == 'Train':
            self._variable_decay(weights,wd)
            self._variable_decay(biases,bd)
        
        return fc
        
        
#    def _Metric_Loss(self,Feature,output):#the half of the batch is Source,the rest is the target
#        batchsize = output.get_shape().as_list()[0]   
##        dim = bottom.get_shape().as_list()[1] 
#        num = tf.cast(0.0, tf.float32)
#        loss = tf.cast(0.0, tf.float32)
#        tem_label = tf.cast(tf.argmax(output ,1),dtype=tf.uint8)
#        def interclass():
#            return -0.1*tf.nn.l2_loss((Feature[i,:]-Feature[j,:]))*2,num+1
#        def intraclass():
#            return tf.nn.l2_loss((Feature[i,:]-Feature[j,:]))*2,num+1
#        for i in range(batchsize-1):
#            for j in range(i+1,batchsize):
#                tem_loss,num = tf.cond(tf.equal(tem_label[i],tem_label[j]),intraclass,interclass)
#                loss += tem_loss
#        loss = loss / (num * 2.0)
#        tf.add_to_collection('losses', loss)
#
#        return loss

                       
    def _variable_decay(self,var,wd):
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name=var.op.name+'_loss')
        tf.add_to_collection('losses', weight_decay)
        return
    
    def _Dropout(self,bottom):
        if self.Phase=='Train':
            bottom = tf.nn.dropout(bottom,keep_prob=0.5)
        else:
            bottom = tf.nn.dropout(bottom,keep_prob=1.0)
        return bottom

    def _pre_process(self,DataBatch):
        rgb_scaled = DataBatch

        # Convert RGB to BGR
        red, green, blue = tf.split(3, 3, rgb_scaled)
        assert red.get_shape().as_list()[1:] == [227, 227, 1]
        assert green.get_shape().as_list()[1:] == [227, 227, 1]
        assert blue.get_shape().as_list()[1:] == [227, 227, 1]
        
        bgr = tf.concat(3, [

            red - Alex_MEAN[0],
            green - Alex_MEAN[1],
            blue - Alex_MEAN[2]
            
        ])
        
        
        assert bgr.get_shape().as_list()[1:] == [227, 227, 3]
        return bgr        

    def build(self, Bottom , Phase , SDA_Tradeoff=0.0):
        self.Phase = Phase
        batchsize = Bottom.get_shape().as_list()[0]
        Bottom = self._pre_process(Bottom)

        
        
        '''The First Scale Convolution Layers'''
        conv1 = self._conv_layer(bottom=Bottom, pad='VALID',
                                 shape=(11,11,3,96),stride=[1,4,4,1],
                                 initializer=(0.01,0),name="conv1",wd=0.0)
                                                            
        relu1 = self._RELU(conv1)
        pool1 = self._max_pool(relu1,name='pool1')
        norm1 = self._LRN(pool1,'norm1')


       
        ''' The Second Convolution Layer'''  
        conv2 = self._conv_layer(bottom=norm1,pad='SAME',
                                 shape=(5,5,48,256),stride=[1,1,1,1],
                                 initializer=(0.01,0),group=2,name="conv2",wd=0.0)
        relu2 = self._RELU(conv2)
        pool2 = self._max_pool(relu2,name='pool2')
        norm2 = self._LRN(pool2,'norm2')

        ''' THE Thrid Convolution Layer''' 
        conv3 = self._conv_layer(bottom=norm2,pad='SAME',
                                shape=(3,3,256,384),stride=[1,1,1,1],
                                initializer=(0.01,0),name="conv3")        
                                                    
        relu3 = self._RELU(conv3)
  
      
        ''' THE Forth Convolution Layer''' 
        conv4 = self._conv_layer(bottom=relu3,pad='SAME',
                                 shape=(3,3,192,384),stride=[1,1,1,1],
                                 initializer=(0.01,0),group=2,name="conv4")         
        
        relu4 = self._RELU(conv4)
                                        
        ''' THE Fifth Convolution Layer''' 
        conv5 = self._conv_layer(bottom=relu4,pad='SAME',
                                 shape=(3,3,192,256),stride=[1,1,1,1],
                                 initializer=(0.01,0),group=2,name="conv5")                                                    
                                                    
        relu5 = self._RELU(conv5)
        pool5 = self._max_pool(relu5,name='pool5')
        
        '''reshape'''
        reshape = self._reshape_conv_to_fc(pool5)
        
        '''Public FC layer1'''
        fc1 = self._fc_layer(bottom = reshape,
                           output_num=4096, name='fc6',
                           dtype=tf.float32,initializer=(0.005,0.1))
                           
        fc1 = self._RELU(fc1)
        drop1 = self._Dropout(fc1)
        '''Public FC layer2'''
        
        fc2 = self._fc_layer(bottom = drop1,
                           output_num=4096, name='fc7',
                           dtype=tf.float32,initializer=(0.005,0.1))
                           
        fc2 = self._RELU(fc2)
        drop2 = self._Dropout(fc2)
        
        '''Public FC layer3'''
        bottleneck = self._fc_layer(bottom = drop2,
                           output_num=256, name='bottle_neck',
                           dtype=tf.float32,initializer=(0.005,0.1),option=True)
        
#        bottleneck_relu = self._RELU(bottleneck)
#        drop_bottleneck = self._Dropout(bottleneck_relu)
#        

        
        '''Domain Classifier Layer1'''
        Dc1=self._fc_layer(bottom = bottleneck,
                           output_num=1024, name='Dc_1',
                           dtype=tf.float32,initializer=(0.01,0.1),option=True)
        
        Dc1 = self._RELU(Dc1)
        Drop_DC1 = self._Dropout(Dc1)
        
        '''Domain Classifier Layers2'''
        Dc2=self._fc_layer(bottom = Drop_DC1,
                           output_num=1024, name='Dc_2',
                           dtype=tf.float32,initializer=(0.01,0.1),option=True)
                           
        Dc2 = self._RELU(Dc2)
        Drop_DC2 = self._Dropout(Dc2)
        
        '''Domain Classifier Output'''
        Dc_output = self._fc_layer(bottom = Drop_DC2,
                           output_num=2, name='Dc_output',
                           dtype=tf.float32,initializer=(0.01,0.1),option=True)

        '''Label Classifier Output'''
        
        Lc_output =  self._fc_layer(bottom = bottleneck,
                           output_num=31, name='L_output',
                           dtype=tf.float32,initializer=(0.01,0.1),option=True)                  
#        if self.Phase=='Train': 
#            Metric_Loss = self._Metric_Loss(bottleneck_relu,Lc_output)
#        
        if self.Phase=='Train':
            
            SDA = self._fc_layer(bottom = Lc_output,
                           output_num=256, name='SDA',
                           dtype=tf.float32,initializer=(0.005,0.1),option=True)
            SDA = self._RELU(SDA)
            tem_bottleneck = tf.stop_gradient(bottleneck)
            SDA_Loss = tf.mul(tf.nn.l2_loss((SDA - tem_bottleneck)), SDA_Tradeoff*(1.0 / (batchsize)) , name='SDA_Loss')
            tf.add_to_collection('losses', SDA_Loss)
            return Dc_output,Lc_output ,SDA_Loss
        else:
            return Dc_output,Lc_output
                           
              
    def Loss(self,Bottom,Labels,Domain_Labels,Phase,condition_entropy_rate,SDA):
        
        labels = tf.cast(Labels, tf.int64)
        domain_labels = tf.cast(Domain_Labels,tf.int64)
        
        Dc_output,Lc_output,SDA_Loss = self.build(Bottom,Phase,SDA_Tradeoff=SDA)
        cross_entropy_1 = tf.nn.sparse_softmax_cross_entropy_with_logits(Dc_output,domain_labels)
        cross_entropy_domain = tf.reduce_mean(cross_entropy_1, name='cross_entropy_domain')
    
    
        batch_size = Labels.get_shape().as_list()[0]
        cross_entropy_2 = tf.cast(0.0, tf.float32)
        num = tf.cast(0.0, tf.float32) 
        def f1():
            return tf.cast(0.0,dtype=tf.float32),num
        def f2():
            return tf.nn.sparse_softmax_cross_entropy_with_logits(Lc_output[i,:],labels[i]),num+1             
        for i in range(batch_size):
            tem , num = tf.cond(tf.equal(domain_labels[i],1),f1,f2)#0:SourceDomain,1:TargetDomain
            cross_entropy_2 += tem
        cross_entropy_labels = tf.mul(cross_entropy_2 , 1.0 / num , name='cross_entropy_labels')                   
        tf.add_to_collection('losses', cross_entropy_labels)   
        num = tf.cast(0.0, tf.float32)
        condition_entropy = tf.cast(0.0, tf.float32)
        
        def f3():
            mask = tf.ones_like(Lc_output[i,:])*1e-4
            softmax = tf.nn.softmax(Lc_output[i,:])
            return tf.reduce_sum((softmax+mask)*(tf.log(softmax+mask))),num+1
        for i in range(batch_size):
            tem,num = tf.cond(tf.equal(domain_labels[i],0),f1,f3)
            condition_entropy+=tem
        condition_entropy =  tf.mul(condition_entropy ,
                                    -1.0 * condition_entropy_rate/num, name='condition_entropy')  
        tf.add_to_collection('losses', condition_entropy)                           
        return cross_entropy_domain,cross_entropy_labels,condition_entropy ,SDA_Loss
    
      

    def Train(self,Bottom,Labels,Domain_Labels,Domain_Tradeoff,
              base_lr,Global_step,condition_entropy_rate,SDA):
        
        cross_entropy_domain,cross_entropy_labels,condition_entropy,SDA_Loss =self.Loss(Bottom,Labels,
                                                             Domain_Labels,Phase='Train',
                                                             condition_entropy_rate=condition_entropy_rate,
                                                             SDA=SDA)
        cross_entropy_domain_ = tf.mul(cross_entropy_domain,Domain_Tradeoff)
        lr = tf.train.exponential_decay(learning_rate = base_lr,
                                global_step=Global_step,decay_steps=10000,decay_rate=0.5,staircase=True)
#        lr = base_lr * tf.pow((1.0+0.001*Global_step),-0.75)                        
        opt=tf.train.MomentumOptimizer(lr, momentum=0.9)        
#        opt = tf.train.AdamOptimizer(base_lr)
        grad_domain=opt.compute_gradients(cross_entropy_domain_)
        grad_labels = opt.compute_gradients(tf.add_n(tf.get_collection('losses'), name='total_loss'))
#        grad_SDALoss = opt.compute_gradients(SDA_Loss)
        def Grad_Process(grad_domain,grad_label,domain_tradeoff):
            grad = []
            for i in range(14):
                tem1 = grad_domain[i][0]
                tem1 = tem1 * -0.1 * domain_tradeoff 
                tem2 = grad_label[i][0]
                tem2 = tem2 * 0.1
                tem = tem1+tem2
                tem_tuple = (tem , grad_domain[i][1])
                grad.append(tem_tuple)
            for i in range(14,16):
                tem1 = grad_domain[i][0]
                tem1 = tem1 * -1.0 * domain_tradeoff
                tem2 = grad_label[i][0]
                tem = tem1 + tem2
                tem_tuple = (tem,grad_domain[i][1])
                grad.append(tem_tuple)
            for i in range(16,22):
                grad.append(grad_domain[i])
#            for i in range(22,24):
#                tem1 = grad_sda[i][0]
#                tem2 = grad_label[i][0]
#                tem = tem1 +tem2
#                tem_tuple = (tem,grad_label[i][1])
#                grad.append(tem_tuple)
#            for i in range(24,26):
#                grad.append(grad_sda[i])
            for i in range(22,26):
                grad.append(grad_label[i])
            return grad
        
        
        grad = Grad_Process(grad_domain,grad_labels,1.0)
        train_op=opt.apply_gradients(grad,global_step=Global_step)
        
        return  train_op,cross_entropy_domain,cross_entropy_labels,condition_entropy,SDA_Loss


    def Accuracy(self,Bottom,Labels,Domain_Labels):
        
        Dc_output,Lc_output = self.build(Bottom,Phase='Test')
        
        correct_prediction = tf.equal(tf.cast(tf.argmax(Lc_output ,1),dtype=tf.uint8), 
                                      tf.cast(Labels,dtype=tf.uint8))
        accuracy_labels = tf.reduce_mean(tf.cast(correct_prediction, "float")) 
        
        correct_prediction_domain = tf.equal(tf.cast(tf.argmax(Dc_output ,1),dtype=tf.uint8), 
                                      tf.cast(Domain_Labels,dtype=tf.uint8))
                                      
        accuracy_domain = tf.reduce_mean(tf.cast(correct_prediction_domain, "float"))                              
        return accuracy_labels,accuracy_domain                     