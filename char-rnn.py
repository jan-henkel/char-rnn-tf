import argparse
import tensorflow as tf
import pickle
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def split_array(array,fractions):
    splits = np.cumsum(fractions)
    assert splits[-1] == 1, "Fractions don't sum to 1"
    return np.split(array,(splits[:-1]*len(array)).astype(int))

def textdata_load(filename,seq_length=50,val_frac=.05,stride=None):

    if stride is None:
        stride = seq_length
    
    with open(filename,'r') as f:
        txt = f.read()
    
    charmap = sorted(list(set(txt)))
    inv_charmap = {c:i for i,c in enumerate(charmap)}

    def decode(a):
        return "".join(list(map(lambda i:charmap[i],a)))

    def encode(s):
        return np.array(list(map(lambda c:inv_charmap[c],s)))


    segments = [encode(txt[i:i+seq_length+1]) for i in range(0,len(txt)-seq_length-1,stride)]
    
    X = np.array([s[:-1] for s in segments])
    y = np.array([s[1:] for s in segments])
    X_train,X_val = split_array(X,[(1-val_frac),val_frac])
    y_train,y_val = split_array(y,[(1-val_frac),val_frac])
    if stride<seq_length:
        X_val=X_val[((seq_length-1)//stride)-1:]
        y_val=y_val[((seq_length-1)//stride)-1:]

    return {'X_train':X_train,
            'y_train':y_train,
            'X_val':X_val,
            'y_val':y_val,
            'charmap':charmap,'inv_charmap':inv_charmap},encode,decode


parser = argparse.ArgumentParser(description="Train a character-level recurrent neural network on an input text")
parser.add_argument('--rnn-type',nargs=1,dest='rnn_type',choices=['lstm','rnn','gru'],default='lstm')
parser.add_argument('--seq-len',dest='seq_length',type=int,default=50)
parser.add_argument('--stride',dest='stride',type=int,default=None)
parser.add_argument('--path',dest='path',type=str,default='./input.txt')
parser.add_argument('--layers',dest='num_layers',type=int,default=2)
parser.add_argument('--layer-norm',dest='layer_norm',type=bool,default=True)
parser.add_argument('--embed-dim',dest='embedding_dim',type=int,default=None)
parser.add_argument('--hidden-dim',dest='hidden_dim',type=int,default=92)
parser.add_argument('--dropout-keep-prob',dest='dropout_keep_prob',type=np.float32,default=1.)
parser.add_argument('--batch-size',dest='batch_size',type=int,default=50)
parser.add_argument('--iter',dest='iterations',type=int,default=15000)
parser.add_argument('--epochs',dest='epochs',type=int)
parser.add_argument('--learning-rate',dest='learning_rate',type=np.float32,default=3e-4)
parser.add_argument('--val-frac',dest='val_frac',type=np.float32,default=.05)
parser.add_argument('--print-every',dest='print_every',type=int,default=1000)
parser.add_argument('--save-dir',dest='save_dir',type=str,default=None)
parser.add_argument('--sample',dest='sample',type=bool,default=False)
parser.add_argument('--sample-len',dest='sample_length',type=int,default=1000)
parser.add_argument('--prime-text',dest='prime_text',type=str,default=' ')
parser.add_argument('--temp',dest='temperature',type=np.float32,default=.5)
parser.add_argument('--restore-last',dest='restore_last_model',type=bool,default=False)

args = parser.parse_args()

if args.save_dir is None:
    args.save_dir = args.path+"_save"

try:
    with open(os.path.join(args.save_dir,'config.pickle'),'rb') as f:
        config = pickle.load(f)
    for v in ['rnn_type','num_layers','layer_norm','seq_length','embedding_dim']:
        setattr(args,v,getattr(config,v))
except:
    os.makedirs(args.save_dir)
with open(os.path.join(args.save_dir,'config.pickle'),'wb') as f:
    pickle.dump(args,f)
        
g = tf.Graph()
sess = tf.InteractiveSession(graph=g)

text_data,encode,decode = textdata_load(args.path,seq_length=args.seq_length,val_frac=args.val_frac,stride=args.stride)

num_train = len(text_data['X_train'])
num_val = len(text_data['X_val'])
vocab_length = len(text_data['charmap'])

print("num_train:",num_train,"num_val:",num_val,"vocab_length:",vocab_length,"args:",args)

with g.as_default():
    with tf.variable_scope('input_layer'):
        x = tf.placeholder(dtype=tf.int32,shape=[None,args.seq_length],name="x")
        y = tf.placeholder(dtype=tf.int32,shape=[None,args.seq_length],name="y")
        init_state = tf.placeholder(dtype=tf.float32,shape=[3],name="init_state")
        lr = tf.placeholder(dtype=tf.float32,shape=[],name="learning_rate")
        temp = tf.placeholder(dtype=tf.float32,name="temperature")
        keep_prob = tf.placeholder(dtype=tf.float32,name="keep_prob")
    if(args.embedding_dim is not None):
        with tf.variable_scope('embedding_layer'):
            w_emb = tf.get_variable(dtype=tf.float32,shape=[vocab_length,args.embedding_dim],name="embedding_weights")
        
        def embed(x):
            return tf.nn.embedding_lookup(w_emb,x)
    else:
        def embed(x):
            return tf.one_hot(depth=vocab_length,indices=x)
    with tf.variable_scope('embedding_layer'):
        x_emb = embed(x)
    with tf.variable_scope('rnn_layer') as rnn_scope:
        if args.layer_norm:
            cells = [tf.contrib.rnn.LayerNormBasicLSTMCell(args.hidden_dim,dropout_keep_prob=keep_prob) for _ in range(args.num_layers)]
        else:
            rnn_type = {'lstm':tf.contrib.rnn.BasicLSTMCell,'rnn':tf.contrib.rnn.BasicRNNCell,'gru':tf.contrib.rnn.GRUCell}
            cells = [tf.contrib.rnn.DropoutWrapper(rnn_type(args.hidden_dim),input_keep_prob=keep_prob) for _ in range(args.num_layers)]
        stacked_lstm = tf.contrib.rnn.MultiRNNCell(cells=cells)
        drnn_out,drnn_final_state = tf.nn.dynamic_rnn(cell=stacked_lstm,inputs=x_emb,scope="drnn",dtype=tf.float32)
        #drnn_out_2,drnn_final_state_2 = tf.nn.dynamic_rnn(cell=stacked_lstm,inputs=x_emb,scope="drnn",initial_state=init_state)
    with tf.variable_scope('output_layer'):
        w_out = tf.get_variable(dtype=tf.float32,shape=[args.hidden_dim,vocab_length],name="output_weights")
        b_out = tf.get_variable(dtype=tf.float32,shape=[vocab_length],name="output_biases")
        scores = tf.tensordot(drnn_out,w_out,axes=([2],[0]))
    with tf.variable_scope('loss'):
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=scores))
        optim = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
    with tf.variable_scope(rnn_scope) as scope:
        counter = tf.constant(value=0,dtype=tf.int32)
        sample_length = tf.placeholder(dtype=tf.int32,shape=[],name="sample_length")
        x_sample_prime_text = tf.placeholder(dtype=tf.int64,shape=[1,None],name="x_sample_prime_text")
        x_sample_prime_text_emb = embed(x_sample_prime_text)
        scope.reuse_variables()
        prime_drnn_out,prime_drnn_state = tf.nn.dynamic_rnn(cell=stacked_lstm,inputs=x_sample_prime_text_emb,dtype=tf.float32,scope="drnn")
        prime_scores = tf.matmul(prime_drnn_out[:,-1,:],w_out)
        sample = tf.multinomial(prime_scores/temp,1)[0]
        samples = tf.TensorArray(dtype=tf.int64,size=sample_length,element_shape=[1],clear_after_read=False)
        samples = samples.write(0,sample)
        state = prime_drnn_state
        
        def body_(counter,samples,state):
            last_sample = samples.read(counter)
            last_sample_emb = embed(last_sample)
            scope.reuse_variables()
            h,new_state = stacked_lstm(inputs=last_sample_emb,state=state,scope="drnn/multi_rnn_cell")
            new_scores = tf.matmul(h,w_out)
            new_sample = tf.multinomial(new_scores/temp,1)[0]
            samples = samples.write(counter+1,new_sample)
            return counter+1,samples,new_state
        _,sample_out,_ = tf.while_loop(cond=lambda t,*_:t<sample_length-1,body=body_,loop_vars=[counter,samples,state])
        sample_out = sample_out.concat()
    init = tf.global_variables_initializer()

sess.run(init)

def get_batch():
    idx = np.random.choice(num_train,args.batch_size)
    return text_data['X_train'][idx],text_data['y_train'][idx]
def print_sample(length,temperature=1.,prime_text=" "):
    r, = sess.run([sample_out],feed_dict={x_sample_prime_text:np.array([encode(prime_text)]),sample_length:length,temp:temperature,keep_prob:1.})
    print(decode(r))

    
with g.as_default():
    saver = tf.train.Saver()
    
def train(iterations=args.iterations,epochs=args.epochs,print_every=args.print_every,learning_rate=args.learning_rate):
    if epochs is not None:
        iterations_per_epoch = max(num_train // args.batch_size,1)
        iterations = iterations_per_epoch*epochs
        print_every = print_every*iterations_per_epoch
    loss_avg = 0
    for i in range(iterations+1):
        X_batch,y_batch = get_batch()
        if i == 0:
            loss_avg, = sess.run([loss],feed_dict={x:X_batch,y:y_batch,keep_prob:args.dropout_keep_prob})
        else:
            loss_,_ = sess.run([loss,optim],feed_dict={x:X_batch,y:y_batch,lr:learning_rate,keep_prob:args.dropout_keep_prob})
            loss_avg += loss_
        if (i % print_every) == 0:
            val_loss, = sess.run([loss],feed_dict={x:text_data['X_val'],y:text_data['y_val'],keep_prob:1.})
            if i==0:
                best_val_loss = val_loss
            else:
                loss_avg /= print_every
                if val_loss<=best_val_loss:
                    best_val_loss = val_loss
                    saver.save(sess,os.path.join(args.save_dir,'best-model'))
                    print("New best")
            print("Iteration ({}/{})".format(i,iterations),"Loss:",loss_avg,"Validation loss:",val_loss,"\n")
            print("Sample:\n----------------------------------------")
            print_sample(length=200,temperature=.7,prime_text=" ")
            print("----------------------------------------\n")
            loss_avg = 0
            saver.save(sess,os.path.join(args.save_dir,'last-model'))

            
try:
    if args.restore_last_model:
        saver.restore(sess,os.path.join(args.save_dir,'last-model'))
    else:
        saver.restore(sess,os.path.join(args.save_dir,'best-model'))
except:
    pass
    
if args.sample is True:
    print_sample(args.sample_length,args.temperature,args.prime_text)
else:
    train()
    saver.save(sess,os.path.join(args.save_dir,'model'))
