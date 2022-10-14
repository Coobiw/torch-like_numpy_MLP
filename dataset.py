import cfg
import struct
import numpy as np


class MNIST_Dataset:
    def __init__(self,standardization = False):
        self.train_data_path = cfg.train_data_path
        self.test_data_path = cfg.test_data_path

        self.train_label_path = cfg.train_label_path
        self.test_label_path = cfg.test_label_path

        self.standardization = standardization
        self.mean = None
        self.std = None

        self.train_data = self.parse_data(path = self.train_data_path,training=True)
        self.test_data = self.parse_data(path = self.test_data_path,training=False)

        self.train_label = self.parse_label(path = self.train_label_path)
        self.test_label  = self.parse_label(path = self.test_label_path)

    def parse_data(self,path,training):
        offset = 0
        data = open(path, "rb").read()  # read the data from the format file
        # print(data)

        # 获取数据头部信息，魔数、图像数、图像行数、列数
        fmt_header = '>iiii'
        magic_number, img_num, row, column = struct.unpack_from(fmt_header, data, offset)
        # print("魔数:{}\t img_num:{}\t row:{}\t column:{}\t".format(magic_number, img_num, row, column))

        offset += struct.calcsize(fmt_header)
        # print(offset)
        fmt_img_header = '>' + str(img_num * row * column) + 'B'
        img_flatten = struct.unpack_from(fmt_img_header, data, offset)
        img_dataset = np.array(img_flatten,dtype='float32').reshape(img_num, row*column)

        # normalization and standardization
        img_dataset = img_dataset / 255.0
        if self.standardization:
            if training:
                self.mean = np.mean(img_dataset).reshape(1, 1)
                self.std = np.std(img_dataset).reshape(1, 1)
            img_dataset = (img_dataset - self.mean) / self.std

        offset += struct.calcsize(fmt_img_header)
        # print(offset)

        # print(img_dataset.shape)
        return img_dataset

    def parse_label(self,path):
        offset = 0
        data = open(path,"rb").read()

        fmt_header = '>ii'
        magic_number,label_num = struct.unpack_from(fmt_header,data,offset)
        # print("magic_number:{}\t label_num:{}".format(magic_number,label_num))

        offset += struct.calcsize(fmt_header)
        # print(offset)
        fmt_label_header = '>' + str(label_num) + 'B'
        label_set = np.array(struct.unpack_from(fmt_label_header,data,offset)).reshape(label_num,1)

        # 转成one-shot编码
        label_one_hot = np.zeros((label_set.shape[0],10),dtype="float32")
        for i in range(label_set.shape[0]):
            label_one_hot[i][label_set[i][0]] = 1.
        offset += struct.calcsize(fmt_label_header)
        # print(offset)

        # print(label_set.shape)

        return label_one_hot

if __name__ == "__main__":
    # dataset1 = MNIST_Dataset(standardization=True)
    dataset1 = MNIST_Dataset()

    import matplotlib.pyplot as plt

    # plt.figure()
    # counter0=0
    # for each in dataset1.train_label[dataset1.train_data.shape[0]-1]:
    #     if each == 0:
    #         counter0+=1
    #     else:
    #         break
    # plt.imshow(dataset1.train_data[dataset1.train_data.shape[0] - 1].reshape(28,28), cmap="gray")
    # plt.text(24, 24, str(counter0),fontsize = 20,color = [1,1,1])
    #
    # plt.figure()
    # counter = 0
    # for each in dataset1.test_label[dataset1.test_data.shape[0]-1]:
    #     if each == 0:
    #         counter+=1
    #     else:
    #         break
    # plt.imshow(dataset1.test_data[dataset1.test_data.shape[0] - 1].reshape(28,28), cmap="gray")
    # plt.text(24, 24, str(counter),fontsize = 20,color = [1,1,1])
    #
    plt.figure()
    counter2 = 0
    print(dataset1.test_label[61])
    for each in dataset1.test_label[61]:
        if each == 0:
            counter2+=1
        else:
            break
    plt.imshow(dataset1.test_data[61].reshape(28,28), cmap="gray")
    plt.text(24, 24, str(counter2), fontsize=20, color=[1, 1, 1])

    plt.figure()
    counter3 = 0
    print(dataset1.test_label[5067])
    for each in dataset1.test_label[5067]:
        if each == 0:
            counter3 += 1
        else:
            break
    plt.imshow(dataset1.test_data[5067].reshape(28, 28), cmap="gray")
    plt.text(24, 24, str(counter3), fontsize=20, color=[1, 1, 1])

    plt.show()
