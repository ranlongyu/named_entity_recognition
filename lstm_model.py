# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dense
from keras_contrib.layers import CRF

import json
import preprocess_data
import numpy as np

train_x, train_y, test_x, test_y, char_dict = preprocess_data.main_process()

train_x = np.array(train_x[:1646700]).reshape([16467, 100])
train_y = np.array(train_y[:1646700]).reshape([16467, 100, 7])
test_x = np.array(test_x[:182900]).reshape([1829, 100])
test_y = np.array(test_y[:182900]).reshape([1829, 100, 7])

def create_model1():
    model = Sequential()
    '''
    input_dim：这是文本数据中词汇的大小。例如，如果你的数据是整数编码为0-10之间的值，则词表的大小将为11个字。
    output_dim：这是嵌入单词的向量空间的大小。它为每个单词定义了该层的输出向量的大小。例如，它可以是32或100甚至更大。
    input_length：这是输入序列的长度，正如你为Keras模型的任何输入层定义的那样。例如，如果你的所有输入文档包含1000个单词，则为1000。
    这里把每一句划分为100个单词。
    '''
    model.add(Embedding(input_dim=len(char_dict), output_dim=10, input_length=100, mask_zero=False))
    '''
    双向循环神经网络，输出节点为50*2=100个，刚好对应100个标签。
    '''
    model.add(Bidirectional(LSTM(units=50)))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

def create_model2():
    model = Sequential()
    model.add(Embedding(input_dim=len(char_dict), output_dim=10, input_length=100, mask_zero=False))
    model.add(Bidirectional(LSTM(units=32, return_sequences=True)))

    crf = CRF(7, sparse_target=False)
    model.add(crf)
    model.compile('adam', loss=crf.loss_function, metrics=[crf.accuracy])

    model.summary()
    return model

if __name__ == '__main__':
    with open('char_dict.json', 'w', encoding="utf-8") as file_object:
        json.dump(char_dict, file_object)
    model = create_model2()
    model.fit(train_x, train_y, batch_size=32, epochs=5, validation_data=[test_x, test_y])
    model.save('model/crf.h5')
