# -*- coding: utf-8 -*-
from lstm_model import create_model2
import numpy as np
import json

if __name__ == '__main__':
    model_pre = create_model2()
    model_pre.load_weights('model/crf.h5')

    with open('char_dict.json', 'r', encoding="utf-8") as file_object:
        char_dict = json.load(file_object)

    while True:
        predict_text = input('请输入100字以下文字：')
        text_list = list(predict_text)
        if len(text_list) >= 100:
            continue
        text_num = []
        for char in text_list:
            if char in char_dict.keys():
                text_num.append(char_dict[char])
            else:
                text_num.append(0)

        text_num.extend([0]*(100-len(text_num)))
        text_num_pro = np.array(text_num).reshape([1, 100])
        result = model_pre.predict(text_num_pro)

        other = np.array([1, 0, 0, 0, 0, 0, 0], float)
        new_result = []
        str = ''
        for i, item in enumerate(result[0][:len(predict_text)]):
            if (item == other).all():
                if str != '':
                    new_result.append(str)
                    str = ''
                continue
            str += predict_text[i]
        print('命名实体为：', new_result)
