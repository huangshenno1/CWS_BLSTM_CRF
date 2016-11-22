import numpy as np
import tensorflow as tf


def set_hyperparams(_args):
    global args
    args = _args

def uniform_wb(n_in, n_out):
    limit = np.sqrt(6./(n_in+n_out))
    w = tf.get_variable('w', [n_in, n_out], initializer=tf.random_uniform_initializer(-limit, limit))
    b = tf.get_variable('b', [n_out], initializer=tf.constant_initializer(0.))
    return w, b

def score(xs, sequence_lengths, max_sequence_length, keep_prob):
    norm_initializer = tf.random_normal_initializer(stddev=0.01)

    # transfer score
    with tf.variable_scope('A'):
        A = tf.get_variable('A', [args.label_size,  args.label_size], initializer=norm_initializer)

    # embedding
    with tf.variable_scope('emb'):
        lookup_table = tf.get_variable('lookup_table', [args.vocabulary_size, args.embedding_size], initializer=norm_initializer)
	c_emb = tf.nn.embedding_lookup(lookup_table, xs)

    # bigram features
    with tf.variable_scope('bigram'):
	c_emb2 = tf.reshape(tf.concat(3, [c_emb[:,:,:-1,:], c_emb[:,:,1:,:]]), [-1, args.window_size-1, args.embedding_size*2])
	w = tf.get_variable('w', [1, args.embedding_size*2, args.embedding_size], initializer=norm_initializer)
	b = tf.get_variable('b', [args.embedding_size], initializer=tf.constant_initializer(0.))
	b_emb = tf.tanh(tf.nn.conv1d(c_emb2, w, 1, 'VALID') + b)
        
    c_emb = tf.reshape(c_emb, [-1, max_sequence_length, args.window_size*args.embedding_size])
    b_emb = tf.reshape(b_emb, [-1, max_sequence_length, (args.window_size-1)*args.embedding_size])
    emb = tf.concat(2, [c_emb, b_emb])

    # dropout
    x_in = tf.nn.dropout(emb, keep_prob)

    # rnn
    with tf.variable_scope('rnn', initializer=norm_initializer):
	rnn_cell = tf.nn.rnn_cell
	fw_cell = rnn_cell.LSTMCell(args.n_rnn_hidden, state_is_tuple=True)
	bw_cell = rnn_cell.LSTMCell(args.n_rnn_hidden, state_is_tuple=True)
	h, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, x_in, dtype=tf.float32, sequence_length=sequence_lengths)
        h = tf.concat(2, [h[0], h[1]])
        h = tf.reshape(h, [-1, args.n_rnn_hidden*2])

    # full-connection
    with tf.variable_scope('fc'):
	w, b = uniform_wb(args.n_rnn_hidden*2, args.label_size)
	s = tf.matmul(h, w) + b
	
    s = tf.reshape(s, [-1, max_sequence_length, args.label_size])
    return s, A
    
def loss(s, ys, seq_len, A):
    log_likelihood, A = tf.contrib.crf.crf_log_likelihood(s, ys, seq_len, A)
    loss = tf.reduce_mean(-log_likelihood)
    return loss, A

def l2_norm():
    l2 = 0
    for param in tf.trainable_variables():
        l2 = l2 + tf.nn.l2_loss(param)
    return l2

def train(loss):
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(args.learning_rate, global_step, args.decay_steps, args.decay_rate, staircase=True)
    optimizer = eval('tf.train.%sOptimizer' % args.optimizer)
    optimizer = optimizer(learning_rate)
    train_op = optimizer.minimize(loss, global_step=global_step)
    #grads_and_vars = optimizer.compute_gradients(loss)
    #clipped_grads_and_vars = [(tf.clip_by_value(g, -1.e10, 1.e10), v) for g, v in grads_and_vars]
    #train_op = optimizer.apply_gradients(clipped_grads_and_vars, global_step=global_step)
    return train_op, global_step

def predict(s, A, seq_len):
    yp = []
    for ss, l in zip(s, seq_len):
        ss = ss[:l]
        y, score = tf.contrib.crf.viterbi_decode(ss, A)
        yp.append(y)
    return yp
	
