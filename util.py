import numpy as np

def split_array(array,fractions):
    splits = np.cumsum(fractions)
    assert splits[-1] == 1, "Fractions don't sum to 1"
    return np.split(array,(splits[:-1]*len(array)).astype(int))

def decode(a,charmap):
    return "".join(list(map(lambda i:charmap[i],a)))

def encode(s,inv_charmap):
    return np.array(list(map(lambda c:inv_charmap[c],s)))

def preprocess(args):

    if args.stride is None:
        args.stride = args.seq_length
    
    with open(args.path,'r') as f:
        txt = f.read()
    
    charmap = sorted(list(set(txt)))
    inv_charmap = {c:np.uint8(i) for i,c in enumerate(charmap)}
    
    segments = [encode(txt[i:i+args.seq_length+1],inv_charmap) for i in range(0,len(txt)-args.seq_length-1,args.stride)]
    
    X = np.array([s[:-1] for s in segments])
    y = np.array([s[1:] for s in segments])
    X_train,X_val = split_array(X,[(1-args.val_frac),args.val_frac])
    y_train,y_val = split_array(y,[(1-args.val_frac),args.val_frac])

    if args.stride<args.seq_length:
        X_val=X_val[((args.seq_length-1)//args.stride)-1:]
        y_val=y_val[((args.seq_length-1)//args.stride)-1:]

    return {'X_train':X_train,
            'y_train':y_train,
            'X_val':X_val,
            'y_val':y_val},charmap,inv_charmap
