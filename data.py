import cPickle as pkl
import numpy as np


class Data:
    
    def __init__(self, args):
	with open('../data/pku_shorten.pkl', 'rb') as fin:
	    self.train_x, self.train_y, self.test_x, self.test_y, self.dic, self.rdic = pkl.load(fin)
	self.shuffle = not args.nshuffle
	self.batch_size = args.batch_size
	self.window_size = args.window_size

        np.random.seed(args.seed)

        if args.max_seq_len is not None:
            remained_idx = []
            for idx, x in enumerate(self.train_x):
                if 1 < len(x) <= args.max_seq_len:
                    remained_idx.append(idx)
            self.train_x = [self.train_x[idx] for idx in remained_idx]
            self.train_y = [self.train_y[idx] for idx in remained_idx]

	if args.toy_data:
	    self.train_x = self.train_x[:args.toy_data_size]
	    self.train_y = self.train_y[:args.toy_data_size]
	    self.test_x = self.test_x[:args.toy_data_size]
	    self.test_y = self.test_y[:args.toy_data_size]
	self.n_train = len(self.train_x)

	self.generate_minibatches()

    def generate_minibatches(self):
	idx_list = np.arange(self.n_train, dtype='int32')
	if self.shuffle:
	    np.random.shuffle(idx_list)
	
	self.minibatches = []
	self.minibatch_idx = 0
	
	idx = 0
	for i in xrange(self.n_train // self.batch_size):
	    self.minibatches.append(idx_list[idx:idx+self.batch_size])
	    idx += self.batch_size
	if idx != self.n_train:
	    self.minibatches.append(idx_list[idx:])

    def next_batch(self):
	if self.minibatch_idx >= len(self.minibatches):
	    self.generate_minibatches()
    
        x = [self.train_x[i] for i in self.minibatches[self.minibatch_idx]]
	y = [self.train_y[i] for i in self.minibatches[self.minibatch_idx]]
	x, y, l, msl = pack(x, y, self.window_size)
	self.minibatch_idx += 1
	return x, y, l, msl

def contextwin(l, win):
    lpadded = win/2 * [1] + l + win/2 * [2]
    out = [lpadded[i:i+win] for i in xrange(len(l))]
    return np.asarray(out, dtype='int32')

def pack(x_raw, y_raw, win):
    x_raw = [[1]+x+[2] for x in x_raw]
    y_raw = [[0]+y+[0] for y in y_raw]
    l = map(len, x_raw)
    msl = max(l)
    n = len(l)
    x = np.zeros([n, msl, win], dtype='int32')
    y = np.full([n, msl], 0, dtype='int32')
    for i, a in enumerate(x_raw):
        x[i,:len(a),:] = contextwin(a, win)
    for i, a in enumerate(y_raw):
        y[i,:len(a)] = a
    return x, y, l, msl


