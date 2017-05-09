import tensorflow as tf
import numpy as np
import os
from util import encode,decode

class Model:

    def __init__(self,args):
        self.charmap = args.charmap
        self.inv_charmap = args.inv_charmap
        vocab_length = len(self.charmap)
        
        self.graph = tf.Graph()
        with self.graph.as_default():

            #placeholders for data, training- and sampling parameters
            with tf.variable_scope('input_layer'):
                self.x = tf.placeholder(dtype=tf.int32,shape=[None,None],name="x")
                self.y = tf.placeholder(dtype=tf.int32,shape=[None,None],name="y")
                self.lr = tf.placeholder(dtype=tf.float32,shape=[],name="learning_rate")
                self.temp = tf.placeholder(dtype=tf.float32,name="temperature")
                self.keep_prob = tf.placeholder(dtype=tf.float32,name="keep_prob")

            #input embedding (one-hot by default)
            if(args.embedding_dim is not None):
                with tf.variable_scope('embedding_layer'):
                    w_emb = tf.get_variable(dtype=tf.float32,shape=[vocab_length,args.embedding_dim],name="embedding_weights")
                def embed(x):
                    return tf.nn.embedding_lookup(w_emb,x)
            else:
                def embed(x):
                    return tf.one_hot(depth=vocab_length,indices=x)
            with tf.variable_scope('embedding_layer'):
                x_emb = embed(self.x)

            #rnn layers
            with tf.variable_scope('rnn_layer') as rnn_scope:
                if args.rnn_type == 'lstm' and args.layer_norm:
                    cells = [tf.contrib.rnn.LayerNormBasicLSTMCell(args.hidden_dim,dropout_keep_prob=self.keep_prob) for _ in range(args.num_layers)]
                else:
                    rnn_cell = {'lstm':tf.contrib.rnn.BasicLSTMCell,'rnn':tf.contrib.rnn.BasicRNNCell,'gru':tf.contrib.rnn.GRUCell}[args.rnn_type]
                    cells = [tf.contrib.rnn.DropoutWrapper(rnn_cell(args.hidden_dim),input_keep_prob=self.keep_prob) for _ in range(args.num_layers)]
                stacked_lstm = tf.contrib.rnn.MultiRNNCell(cells=cells)
                drnn_out,drnn_final_state = tf.nn.dynamic_rnn(cell=stacked_lstm,inputs=x_emb,scope="drnn",dtype=tf.float32)

            #output layer
            with tf.variable_scope('output_layer'):
                w_out = tf.get_variable(dtype=tf.float32,shape=[args.hidden_dim,vocab_length],name="output_weights")
                b_out = tf.get_variable(dtype=tf.float32,shape=[vocab_length],name="output_biases")
                scores = tf.tensordot(drnn_out,w_out,axes=([2],[0]))

            #loss
            with tf.variable_scope('loss'):
                self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y,logits=scores))
                self.optim = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

            #sampling loop
            with tf.variable_scope(rnn_scope) as scope:
                counter = tf.constant(value=0,dtype=tf.int32)
                self.sample_length = tf.placeholder(dtype=tf.int32,shape=[],name="sample_length")
                self.x_sample_prime_text = tf.placeholder(dtype=tf.int64,shape=[1,None],name="x_sample_prime_text")
                x_sample_prime_text_emb = embed(self.x_sample_prime_text)
                scope.reuse_variables()
                prime_drnn_out,prime_drnn_state = tf.nn.dynamic_rnn(cell=stacked_lstm,inputs=x_sample_prime_text_emb,dtype=tf.float32,scope="drnn")
                prime_scores = tf.matmul(prime_drnn_out[:,-1,:],w_out)
                sample = tf.multinomial(prime_scores/self.temp,1)[0]
                samples = tf.TensorArray(dtype=tf.int64,size=self.sample_length,element_shape=[1],clear_after_read=False)
                samples = samples.write(0,sample)
                state = prime_drnn_state

                def body_(counter,samples,state):
                    last_sample = samples.read(counter)
                    last_sample_emb = embed(last_sample)
                    scope.reuse_variables()
                    h,new_state = stacked_lstm(inputs=last_sample_emb,state=state,scope="drnn/multi_rnn_cell")
                    new_scores = tf.matmul(h,w_out)
                    new_sample = tf.multinomial(new_scores/self.temp,1)[0]
                    samples = samples.write(counter+1,new_sample)
                    return counter+1,samples,new_state
                
                _,sample_out,_ = tf.while_loop(cond=lambda t,*_:t<self.sample_length-1,body=body_,loop_vars=[counter,samples,state])
                self.sample_out = sample_out.concat()
                
            init = tf.global_variables_initializer()
            
            self.sess = tf.Session(graph=self.graph)
            self.sess.run(init)

            self.saver = tf.train.Saver()

    def print_sample(self,length,temperature=1.,prime_text=" "):
        r = self.sess.run(self.sample_out,feed_dict={self.x_sample_prime_text:np.array([encode(prime_text,self.inv_charmap)]),self.sample_length:length,self.temp:temperature,self.keep_prob:1.})
        print(decode(r,self.charmap))

    def restore(self,save_args):
        if save_args.restore_last:
            self.saver.restore(self.sess,os.path.join(save_args.save_dir,'last-model'))
        else:
            self.saver.restore(self.sess,os.path.join(save_args.save_dir,'best-model'))
        
    def train(self,data,train_args,save_args):
        num_train = len(data['X_train'])
        
        def get_batch():
            idx = np.random.choice(num_train,train_args.batch_size)
            return data['X_train'][idx],data['y_train'][idx]
        
        loss_avg = 0
        for i in range(train_args.iterations+1):
            X_batch,y_batch = get_batch()
            if i == 0:
                loss_avg = self.sess.run(self.loss,feed_dict={self.x:X_batch,self.y:y_batch,self.keep_prob:train_args.dropout_keep_prob})
            else:
                loss_,_ = self.sess.run([self.loss,self.optim],feed_dict={self.x:X_batch,self.y:y_batch,self.lr:train_args.learning_rate,self.keep_prob:train_args.dropout_keep_prob})
                loss_avg += loss_
            if (i % train_args.print_every) == 0 or i==train_args.iterations:
                val_loss = self.sess.run(self.loss,feed_dict={self.x:data['X_val'],self.y:data['y_val'],self.keep_prob:1.})
                if i==0:
                    best_val_loss = val_loss
                else:
                    loss_avg /= train_args.print_every
                    if val_loss<=best_val_loss:
                        best_val_loss = val_loss
                        self.saver.save(self.sess,os.path.join(save_args.save_dir,'best-model'))
                        print("Validation loss improved")
                print("Iteration ({}/{})".format(i,train_args.iterations),"Loss:",loss_avg,"Validation loss:",val_loss,"\n")
                print("Sample:\n----------------------------------------")
                self.print_sample(length=200,temperature=.7,prime_text=" ")
                print("----------------------------------------\n")
                loss_avg = 0
                self.saver.save(self.sess,os.path.join(save_args.save_dir,'last-model'))
