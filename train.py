import argparse
import pickle
import numpy as np
import os
from util import preprocess
from model import Model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #turn off warnings

#input preprocessing settings
input_parser = argparse.ArgumentParser(add_help=False)
input_parser.add_argument('path',metavar='PATH',type=str,default='./input.txt',help='path of the input file')
input_group=input_parser.add_argument_group('input preprocessing')
input_group.add_argument('--seq-len',dest='seq_length',type=int,default=50,help='sequence length')
input_group.add_argument('--stride',dest='stride',type=int,default=None,help='stride, defaults to sequence length')
input_group.add_argument('--val-frac',dest='val_frac',type=np.float32,default=.05,help='fraction of data used for validation set') 
input_group.add_argument('--reprocess',dest='reprocess',action='store_const',const=True,default=False,help='do preprocessing again (otherwise preprocessing arguments will be ignored if preprocessed data is already present)')

#rnn model settings
model_parser = argparse.ArgumentParser(add_help=False)
model_group=model_parser.add_argument_group('model parameters')
model_group.add_argument('--type',dest='rnn_type',choices=['lstm','rnn','gru'],default='lstm',help='rnn type')
model_group.add_argument('--layers',dest='num_layers',type=int,default=2,help='number of layers')
model_group.add_argument('--layer-norm',dest='layer_norm',type=bool,default=False,help='use layer normalization. has no effect on anything other than lstm')
model_group.add_argument('--embed-dim',dest='embedding_dim',type=int,default=None,help='embedding dimension. defaults to one-hot encoding if not specified')
model_group.add_argument('--hidden-dim',dest='hidden_dim',type=int,default=92,help='rnn hidden layer dimension')

#model save and restore settings
save_parser = argparse.ArgumentParser(add_help=False)
save_group=save_parser.add_argument_group('model save and restore settings')
save_group.add_argument('--save-dir',dest='save_dir',type=str,default=None,help='optional, defaults to path derived from input file')
save_group.add_argument('--restore-last',dest='restore_last',action='store_const',const=True,default=False,help='restore last model rather than best performing model. does nothing if no previous model is present')
save_group.add_argument('--clear-model',dest='clear',action='store_const',const=True,default=False,help='clear previous model parameters')

#optimization settings
train_parser = argparse.ArgumentParser(add_help=False)
train_group=train_parser.add_argument_group('training parameters')
train_group.add_argument('--iter',dest='iterations',type=int,default=10000,help='number of iterations')
train_group.add_argument('--lr',dest='learning_rate',type=np.float32,default=1e-3,help='learning rate')
train_group.add_argument('--batch-size',dest='batch_size',type=int,default=50,help='batch size')
train_group.add_argument('--dropout',dest='dropout_keep_prob',type=np.float32,default=1.,help='dropout keep probability')
train_group.add_argument('--print-every',dest='print_every',type=int,default=1000,help='number of iterations between progress reports and checkpoints')

help_message_parser = argparse.ArgumentParser(description="Train a character-level recurrent neural network on an input text",parents=[input_parser,model_parser,save_parser,train_parser])
help_message_parser.parse_args()

input_args,rest = input_parser.parse_known_args()
model_args,rest = model_parser.parse_known_args(rest)
save_args,rest = save_parser.parse_known_args(rest)
train_args = train_parser.parse_args(rest)

if save_args.save_dir is None:
    save_args.save_dir = os.path.splitext(input_args.path)[0]+"_save"

try:
    #restore model
    if not save_args.clear:
        with open(os.path.join(save_args.save_dir,'model.pickle'),'rb') as f:
            model_args = pickle.load(f)
    else:
        [os.remove(os.path.join(save_args.save_dir,x)) for x in os.listdir(save_args.save_dir) if 'model' in x or 'checkpoint' in x]
    model = Model(model_args)
    try:
        model.restore(save_args)
    except:
        pass
    
    #load preprocessed data or generate it
    if not input_args.reprocess:
        with open(os.path.join(save_args.save_dir,'data.pickle'),'rb') as f:
            data = pickle.load(f)
    else:
        data,_,_ = preprocess(input_args)
except:
    try:
        os.makedirs(save_args.save_dir)
    except:
        pass
    #preprocess input and set encoding and decoding routines for the model
    data,model_args.charmap,model_args.inv_charmap = preprocess(input_args)

    #create model
    model = Model(model_args)
    
    #save processed data and model
    with open(os.path.join(save_args.save_dir,'model.pickle'),'wb') as f:
        pickle.dump(model_args,f)
    
    with open(os.path.join(save_args.save_dir,'data.pickle'),'wb') as f:
        pickle.dump(data,f)
        
model.train(data,train_args,save_args)
