import argparse
import pickle
import numpy as np
import os
from model import Model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #turn off warnings

#input settings
input_parser = argparse.ArgumentParser(add_help=False)
input_parser.add_argument('path',type=str,default='./input.txt',help='path of the input file')

#model save and restore settings
save_parser = argparse.ArgumentParser(add_help=False)
save_group=save_parser.add_argument_group('model restore settings')
save_group.add_argument('--save-dir',dest='save_dir',type=str,default=None,help='model save directory, defaults to path derived from input file')
save_group.add_argument('--restore-last',dest='restore_last',action='store_const',const=True,default=False,help='restore last model rather than best performing model')

#sample settings
sample_parser = argparse.ArgumentParser(add_help=False)
sample_group=sample_parser.add_argument_group('sample settings')
sample_group.add_argument('--len',dest='sample_length',type=int,default=1000,help='sample length')
sample_group.add_argument('--prime-text',dest='prime_text',type=str,default=' ',help='prime the rnn with an input')
sample_group.add_argument('--temp',dest='temperature',type=np.float32,default=.5,help='sampling temperature, smaller values lead to more conservative guesses')

help_message_parser = argparse.ArgumentParser(description="Sample from a previously trained RNN model",parents=[input_parser,save_parser,sample_parser])
help_message_parser.parse_args()

input_args,rest = input_parser.parse_known_args()
save_args,rest = save_parser.parse_known_args(rest)
sample_args = sample_parser.parse_args(rest)

if save_args.save_dir is None:
    save_args.save_dir = os.path.splitext(input_args.path)[0]+"_save"

try:
    #restore model
    with open(os.path.join(save_args.save_dir,'model.pickle'),'rb') as f:
        model_args = pickle.load(f)
    model = Model(model_args)
    try:
        model.restore(save_args)
    except:
        pass
except:
    raise Exception('No model present')

model.print_sample(sample_args.sample_length,sample_args.temperature,sample_args.prime_text)
