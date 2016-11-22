import time
import sys
import argparse
import logging
import random
import cPickle as pkl
import tensorflow as tf
import numpy as np
import data
import model
from utils import *


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(message)s', filename='log', filemode='w') 
    
    parser = argparse.ArgumentParser()
    
    # data
    parser.add_argument('-toy', '--toy_data', action='store_true')
    parser.add_argument('-ts', '--toy_data_size', type=int, default=1000)
    parser.add_argument('--nshuffle', action='store_true')
    parser.add_argument('-bs', '--batch_size', type=int, default=10)
    parser.add_argument('-msl', '--max_seq_len', type=int, default=60) # pku >= 54
    
    # train
    parser.add_argument('-n', '--n_epochs', type=int, default=100)
    parser.add_argument('-q', '--quiet', action='store_true')
    parser.add_argument('-opt', '--optimizer', default='Adam')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01)
    parser.add_argument('-dr', '--decay_rate', type=float, default=0.95)
    parser.add_argument('-ds', '--decay_steps', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=-1)

    # model
    parser.add_argument('--label_size', type=int, default=4)
    parser.add_argument('-ws', '--window_size', type=int, default=5)
    parser.add_argument('-vs', '--vocabulary_size', type=int, default=4554) # pku 4554
    parser.add_argument('-es', '--embedding_size', type=int, default=50)
    parser.add_argument('-nh', '--n_rnn_hidden', type=int, default=150)
    
    # norm
    parser.add_argument('-l2', '--l2_coefficient', type=float, default=0.0001)
    parser.add_argument('-kp', '--keep_prob', type=float, default=0.8)

    # save args
    args = parser.parse_args()
    
    if args.seed < 0:
        args.seed = random.randint(0, 4294967295)

    logging.info(args)
    if not args.quiet:
        print args
   
    with open('hyperparams.pkl', 'wb') as fout:
	pkl.dump(args, fout)
     
    # load data
    dataset = data.Data(args)

    # train
    tic0 = time.time()
    model.set_hyperparams(args)
    with tf.Graph().as_default():
        # random seed    
        tf.set_random_seed(args.seed)

	x_= tf.placeholder(tf.int32, [None, None, args.window_size])
	y_= tf.placeholder(tf.int32, [None, None])
	seq_len_= tf.placeholder(tf.int64, [None])
        msl_ = tf.placeholder(tf.int32, [])
        keep_prob_ = tf.placeholder(tf.float32, [])

	s_, A_ = model.score(x_, seq_len_, msl_, keep_prob_)
        
	c_, A_ = model.loss(s_, y_, seq_len_, A_)
        l2_ = args.l2_coefficient * model.l2_norm()
        loss_ = c_ + l2_

	train_op, global_step = model.train(loss_)

	logging.info('\n'+'='*50)
	logging.info('Trainable Parameters')
	for param in tf.trainable_variables():
	    logging.info('  '+param.name)
            # print sess.run([param])
	logging.info('='*50+'\n')
        saver = tf.train.Saver(tf.trainable_variables())
    
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True

	sess = tf.Session(config=config)
	init = tf.initialize_all_variables()
	sess.run(init)
        logging.info('Model builded, %s used\n' % time_format(time.time() - tic0))
        if not args.quiet:
            print('Model builded, %s used\n' % time_format(time.time() - tic0))

        # pre-store test dataset feed_dicts
        test_feed_dicts = []
        for i in xrange((len(dataset.test_x)+args.batch_size-1) // args.batch_size):
            x, y, l, msl = data.pack(
                dataset.test_x[i*args.batch_size:(i+1)*args.batch_size], 
                dataset.test_y[i*args.batch_size:(i+1)*args.batch_size], 
                args.window_size)
            test_feed_dicts.append({x_:x, seq_len_:l, y_:y, msl_:msl, keep_prob_:1.})
        
        for epoch in xrange(1, args.n_epochs+1):
            tic = time.time()
	    loss = 0.
            n_train = dataset.n_train
            n_trained = 0
	    for idxs in dataset.minibatches:
	        x, y, l, msl = dataset.next_batch()
                _, c = sess.run([train_op, loss_], feed_dict={x_:x, seq_len_:l, y_:y, msl_:msl, keep_prob_:args.keep_prob})
                if np.isnan(c):
                    logging.error('Gradient Explosion!')
                    print('Gradient Explosion!')
                    exit()
                n_trained += len(idxs)
	        loss += c
                if not args.quiet:
		    print '[training] epoch %i >> %2.2f%% [%f], completed in %s << \r' % (epoch, n_trained*100./n_train, c, time_format((time.time() - tic) * (n_train-n_trained) / n_trained)), 
                    sys.stdout.flush()

            loss = loss / len(dataset.minibatches)
	    logging.info('[training] epoch %i >> loss = %f , %s [%s] used' % (epoch, loss, time_format(time.time() - tic), time_format(time.time() - tic0)))
            if not args.quiet:
	        print('[training] epoch %i >> loss = %f , %s [%s] used' % (epoch, loss, time_format(time.time() - tic), time_format(time.time() - tic0)))
	    
            if epoch % 5 == 0:
                saver.save(sess, 'models/model', global_step=epoch)
            
            pred_y = []
            for test_feed_dict in test_feed_dicts:
                s, A = sess.run([s_, A_], feed_dict=test_feed_dict)
                l = test_feed_dict[seq_len_]
                pred_y_i = model.predict(s, A, l)
                pred_y.extend(pred_y_i)
	    p, r, f = evaluate(pred_y, dataset.test_y)
	    logging.info('[testing]  P: %2.2f%% R: %2.2f%% F: %2.2f%%' % (p*100., r*100., f*100.))
            if not args.quiet:
		rd = random.randrange(len(pred_y))
                print pred_y[rd][1:-1]
                print dataset.test_y[rd]
	        print('[testing]  P: %2.2f%% R: %2.2f%% F: %2.2f%%' % (p*100., r*100., f*100.))
