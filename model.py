import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import legacy_seq2seq

import numpy as np


class Model():
    def __init__(self, args, training=True):
        self.args = args
        if not training:
            args.batch_size = 1
            args.seq_length = 1

        if args.model == 'rnn':
            cell_fn = rnn.BasicRNNCell
        elif args.model == 'gru':
            cell_fn = rnn.GRUCell
        elif args.model == 'lstm':
            cell_fn = rnn.BasicLSTMCell
        elif args.model == 'nas':
            cell_fn = rnn.NASCell
        else:
            raise Exception("model type not supported: {}".format(args.model))

        cells = []
        for _ in range(args.num_layers):
            cell = cell_fn(args.rnn_size)   #여기서의 rnn_size는 out_put layer의 크기라고 볼수 있다??
            if training and (args.output_keep_prob < 1.0 or args.input_keep_prob < 1.0):
                cell = rnn.DropoutWrapper(cell,
                                          input_keep_prob=args.input_keep_prob,
                                          output_keep_prob=args.output_keep_prob)
            cells.append(cell)

        self.cell = cell = rnn.MultiRNNCell(cells, state_is_tuple=True)

        self.input_data = tf.placeholder(   # embedding을 위함
            tf.int32, [args.batch_size, args.seq_length])
        self.targets = tf.placeholder(  #정답 값을 가지고 있는 데이터
            tf.int32, [args.batch_size, args.seq_length])
        self.initial_state = cell.zero_state(args.batch_size, tf.float32) #3*2*60*700

        with tf.variable_scope('rnnlm'):
            softmax_w = tf.get_variable("softmax_w",    #w개수만큼 생성
                                        [args.rnn_size, args.vocab_size])
            softmax_b = tf.get_variable("softmax_b", [args.vocab_size]) #b개수만큼 생성

        embedding = tf.get_variable("embedding", [args.vocab_size, args.rnn_size])  #train을 통해 char 데이터에 해당하는 수치들을 갱신하면서 embedding table을 갱신하고
        inputs = tf.nn.embedding_lookup(embedding, self.input_data) #batch_size 만큼의 데이터를 각각의 데이터에 해당하는 embedding vec로 변환해서 inputs변수에 저장 60*100*700

        # dropout beta testing: double check which one should affect next line
        if training and args.output_keep_prob:
            inputs = tf.nn.dropout(inputs, args.output_keep_prob)

        inputs = tf.split(inputs, args.seq_length, 1)  #60*100*(700) => 100*60*1*(700)
        inputs = [tf.squeeze(input_, [1]) for input_ in inputs] #100*60*1*(700) => 100*60*(700)

        def loop(prev, _):  #train이 아니고 과거 훈련 모델로 값을 출력하고 싶은경우 더이상 bp를 할 필요가 없기에 설정해주는 것 같음.
            prev = tf.matmul(prev, softmax_w) + softmax_b
            prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
            return tf.nn.embedding_lookup(embedding, prev_symbol)

        outputs, last_state = legacy_seq2seq.rnn_decoder(inputs, self.initial_state, cell, loop_function=loop if not training else None, scope='rnnlm')
        #outputs는 100*60*700 last_state는 3*2*60*700 inputs를 lstm의 입력으로 전달
        output = tf.reshape(tf.concat(outputs, 1), [-1, args.rnn_size]) #6000*700


        self.logits = tf.matmul(output, softmax_w) + softmax_b #6000*700 x 700*voca_size + voca_size
        self.probs = tf.nn.softmax(self.logits) #가중치 및 바이어스 값들 계산 및 softmax
        loss = legacy_seq2seq.sequence_loss_by_example( #rnn에 특화된 loss fanction, loss(erorr)를 한번에 계산
                [self.logits],  #Y에 해당하는 예측값
                [tf.reshape(self.targets, [-1])],   #정답 레이블
                [tf.ones([args.batch_size * args.seq_length])]) #가중치
        self.cost = tf.reduce_sum(loss) / args.batch_size / args.seq_length #최종 cost계산
        with tf.name_scope('cost'):
            self.cost = tf.reduce_sum(loss) / args.batch_size / args.seq_length
        self.final_state = last_state   #cell의 최종 상태
        tvars = tf.trainable_variables() #가시적인 그래프 생성
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                args.grad_clip) #그래디언트의 크기를 줄이기 위함
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer()    #adamoptimizer 사용
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))    #그래디언트 클리프를 적용하여 학습하도록 함.

        # instrument tensorboard
        tf.summary.histogram('logits', self.logits)
        tf.summary.histogram('loss', loss)
        tf.summary.scalar('train_loss', self.cost)

    def sample(self, sess, chars, vocab, num=200, prime=' ', sampling_type=1):  #학습된 모델로 출력하기 위함.
        """
        prime: starting character sequence.
        sampling_type: 0 to use max at each timestep, 1 to sample at each timestep, 2 to sample on spaces
        """
        state = sess.run(self.cell.zero_state(1, tf.float32))   #state값에 0 채우기
        print('Generate sample start with:', prime)
        # make rnn state by feeding in prime sequence.
        for char in prime[:-1]: #prime(seed 문자열)을 한글자씩 입력
            print('put this to rnn to make state:', char)
            x = np.zeros((1, 1)) # 1x1 matrix
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state: state}
            [state] = sess.run([self.final_state], feed)

        def weighted_pick(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            return(int(np.searchsorted(t, np.random.rand(1)*s)))

        ret = prime
        char = prime[-1]
        for n in range(num):
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]   #key 값(vocab 글자)에 의한 index값
            feed = {self.input_data: x, self.initial_state: state}
            [probs, state] = sess.run([self.probs, self.final_state], feed)
            p = probs[0]

            if sampling_type == 0:
                sample = np.argmax(p)
            elif sampling_type == 2:
                if char == ' ':
                    sample = weighted_pick(p)
                else:
                    sample = np.argmax(p)
            else:  # sampling_type == 1 default:
                sample = weighted_pick(p)

            pred = chars[sample]
            ret += pred
            char = pred
        return ret