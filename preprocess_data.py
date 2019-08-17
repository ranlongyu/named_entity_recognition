# -*- coding: utf-8 -*-
import re

file_path = "./ChineseCorpus199801.txt"
train = []
train_tag = []
char_dict = {}

# 给单词中的每一个字符打标签
# 其他字符：1，人名（nr）首字：2，人名非首字：3，地名（ns）首字：4，地名非首字：5，组织名（nt）首字：6，组织名非首字：7
def encode(word, tag):
    train.extend(list(word))
    if tag != 'nr' and tag != 'ns' and tag != 'nt':
        train_tag.extend([[1, 0, 0, 0, 0, 0, 0]]*len(list(word)))
    elif tag == 'nr':
        train_tag.append([0, 1, 0, 0, 0, 0, 0])
        train_tag.extend([[0, 0, 1, 0, 0, 0, 0]]*(len(list(word))-1))
    elif tag == 'ns':
        train_tag.append([0, 0, 0, 1, 0, 0, 0])
        train_tag.extend([[0, 0, 0, 0, 1, 0, 0]] * (len(list(word)) - 1))
    elif tag == 'nt':
        train_tag.append([0, 0, 0, 0, 0, 1, 0])
        train_tag.extend([[0, 0, 0, 0, 0, 0, 1]] * (len(list(word)) - 1))

# 将文本转化为字符，标签的形式
def process():
    with open(file_path, encoding='GBK') as file:
        for line in file:
            if line == '\n':
                continue
            line = line.split()
            del line[0]
            word_join = ''
            word_middle = 0
            name_join = ''
            name_middle = 0
            for word in line:
                result = re.match('^(\S+)/(\S+)', word)
                # 前三个if处理如“[中央/n 人民/n 广播/vn 电台/n]nt”的格式
                begin = re.match('^\[(\S+)', result.group(1))
                if begin:
                    word_join += begin.group(1)
                    word_middle = 1
                    continue
                end = re.match('^(\S+)](\S+)', result.group(2))
                if end:
                    word_join += result.group(1)
                    encode(word_join, end.group(2))
                    word_middle = 0
                    word_join = ''
                    continue
                if word_middle == 1:
                    word_join += result.group(1)
                    continue
                # 这个if处理姓、名分开的情况，如“邓/nr 小平/nr”
                if result.group(2) == 'nr':
                    name_join += result.group(1)
                    if name_middle == 1:
                        encode(name_join, result.group(2))
                        name_middle = 0
                        name_join = ''
                        continue
                    name_middle = 1
                    continue
                name_middle = 0
                encode(result.group(1), result.group(2))

# 将字符转化为数字，并切分训练集与测试集
def pro_process():
    j = 0
    for i, char in enumerate(train):
        if char not in char_dict.keys():
            char_dict[char] = j
            j = j + 1
        train[i] = char_dict[char]
    divide = int(len(train) * 0.9)
    global test
    test = train[divide:]
    del train[divide:]
    global test_tag
    test_tag = train_tag[divide:]
    del train_tag[divide:]

# 运行程序，依次返回训练集，训练集标签，测试集，测试集标签，字符字典
def main_process():
    process()
    pro_process()
    return train, train_tag, test, test_tag, char_dict

if __name__ == '__main__':
    train, train_tag, test, test_tag, char_dict = main_process()
    print(train)
    #print(train_tag)
    print(len(train))
    print(len(train_tag))
    print(char_dict)
