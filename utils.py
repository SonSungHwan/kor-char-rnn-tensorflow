import codecs
import os
import collections
from six.moves import cPickle
import numpy as np


class TextLoader():
    def __init__(self, data_dir, batch_size, seq_length, encoding='utf-8'):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.encoding = encoding

        input_file = os.path.join(data_dir, "input.txt")
        vocab_file = os.path.join(data_dir, "vocab.pkl")
        tensor_file = os.path.join(data_dir, "data.npy")

        if not (os.path.exists(vocab_file) and os.path.exists(tensor_file)):
            print("reading text file")
            self.preprocess(input_file, vocab_file, tensor_file)
        else:
            print("loading preprocessed files")
            self.load_preprocessed(vocab_file, tensor_file)
        self.create_batches()
        self.reset_batch_pointer()

    def preprocess(self, input_file, vocab_file, tensor_file):
        with codecs.open(input_file, "r", encoding=self.encoding) as f:
            data = f.read()
        counter = collections.Counter(data) #data를 char 기준으로 카운트 리스트 생성
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])  #카운트 리스트 1번째(카운트 횟수)기준으로 소팅
        self.chars, _ = zip(*count_pairs)   #count_pairs 의 0번째 인자들(글자)는 chars에 횟수 인자들은 _에 저장
        self.vocab_size = len(self.chars) #추출한 글자의 종류 개수
        self.vocab = dict(zip(self.chars, range(len(self.chars)))) #chars에 index부여
        with open(vocab_file, 'wb') as f:
            cPickle.dump(self.chars, f) #chars를 파일로 저장
        self.tensor = np.array(list(map(self.vocab.get, data))) #data의 글자들을 vocab index 값들로 치환
        np.save(tensor_file, self.tensor)

    def load_preprocessed(self, vocab_file, tensor_file):
        with open(vocab_file, 'rb') as f:
            self.chars = cPickle.load(f)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.tensor = np.load(tensor_file)
        self.num_batches = int(self.tensor.size / (self.batch_size *
                                                   self.seq_length))

    def create_batches(self):
        self.num_batches = int(self.tensor.size / (self.batch_size *
                                                   self.seq_length))    #공백을 포함한 data문자 수/(60*100) @back propagation의 횟수와 동일

        # When the data (tensor) is too small,
        # let's give them a better error message
        if self.num_batches == 0:
            assert False, "Not enough data. Make seq_length and batch_size small."

        self.tensor = self.tensor[:self.num_batches * self.batch_size * self.seq_length] #num_batches값에 딱 떨어지게 잘라냄
        xdata = self.tensor
        ydata = np.copy(self.tensor)
        ydata[:-1] = xdata[1:]
        ydata[-1] = xdata[0]    #ydata의 데이터를 한칸 밀어서 생성
        self.x_batches = np.split(xdata.reshape(self.batch_size, -1),   #xdata.len/batch_size 의 원소의 개수로 batch_size로 분할
                                  self.num_batches, 1)  #ex)12000개의 글자 data라면, reshape로 60*200의 데이터를 생성하고
                                                        #split으로 2*60*100의 데이터로 만듬 즉 한덩어리씩 2번 bp를 진행
        self.y_batches = np.split(ydata.reshape(self.batch_size, -1),
                                  self.num_batches, 1)

    def next_batch(self):   #배치 포인터
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return x, y

    def reset_batch_pointer(self):
        self.pointer = 0

