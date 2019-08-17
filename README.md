# 命名实体识别

## 需要安装的包

- python3
- tensorflow
- keras
    - keras 中的 CRF 层需要安装 keras-contrib，可以参考[keras-contrib][1]
- re
- numpy
- json

## 运行说明 

文件夹中包含了已经训练好的模型，直接运行命令行即可进行测试：

    python application.py

如果想自己训练模型可以：

    python lstm_model.py


  [1]: https://github.com/keras-team/keras-contrib
