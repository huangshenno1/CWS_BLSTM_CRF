def evaluate(pred_y, y, beta = 1.0):
    true_word_count = 0
    test_word_count = 0
    right = wrong = 0
    for y1, y2 in zip(pred_y, y):
        y1 = y1[1:-1]
	y1 = label2len(y1)
	y2 = label2len(y2)
        assert len(y1) == len(y2)
	for i in xrange(len(y1)):
	    if y1[i]: test_word_count += 1
	    if y2[i]: true_word_count += 1
	    if y2[i] > 0 and y1[i] == y2[i]: right += 1
    if right == 0:
	p = r = f = 0.0
    else:
	p = 1.0 * right / test_word_count
	r = 1.0 * right / true_word_count
	f = (1+beta)*p*r / (beta*p+r)
    return p, r, f

def label2len(label):
    out = [0] * len(label)
    l = 0
    for i in xrange(len(label)-1, -1, -1):
	l += 1
	if label[i] == 0 or label[i] == 1:
	    out[i] = l
	    l = 0
    return out

def time_format(t):
    s = ''
    if t >= 3600: 
	s += '%d h ' % (t/3600)
	t %= 3600
    if t >= 60:
	s += '%d m ' % (t/60)
	t %= 60
    s += '%d s' % t
    return s

