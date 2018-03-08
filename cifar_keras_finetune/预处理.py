import os
import shutil
import pandas as pd

import datetime

data_dir = '/home/jiang/Desktop/workspace/cifar-10/'
train_dir = data_dir + 'train_5000'
input_dir = data_dir + 'input_5000'
test_dir = data_dir + 'test_1000'
label_file = 'trainLabels.csv'

############## 预处理 ####################
idx_label_pd = pd.read_csv(label_file)
idx_label_pd.head()
# 将 label 存储在一个集合之中， 并输出这个集合
labels = set(idx_label_pd['label'])
print(labels)

# 通过 listdir 将文件存储在列表中，并读取列表长度，即文件的个数
num_train = len(os.listdir(os.path.join(data_dir, train_dir)))
print(num_train)

idx_label_train = idx_label_pd[:45000]
idx_label_test = idx_label_pd[45000:50000]


# 创建文件夹
def rmdir_then_mkdir(dirpath):
	if os.path.exists(dirpath):
		shutil.rmtree(dirpath)
	os.mkdir(dirpath)


rmdir_then_mkdir(input_dir)

# 5000张训练，1000张测试
num_train_per_category = 4000
total_train = num_train_per_category * 10

num_test_per_category = 100
total_test = num_test_per_category * 10


# 将训练集和测试集的文件移至对应文件夹
def mv_file(idx_label, num_per_category, aim_dir):
	# 创建10个类别文件夹
	frog_dir = os.path.join(aim_dir, 'frog')
	rmdir_then_mkdir(frog_dir)

	automobile_dir = os.path.join(aim_dir, 'automobile')
	rmdir_then_mkdir(automobile_dir)

	horse_dir = os.path.join(aim_dir, 'horse')
	rmdir_then_mkdir(horse_dir)

	bird_dir = os.path.join(aim_dir, 'bird')
	rmdir_then_mkdir(bird_dir)

	cat_dir = os.path.join(aim_dir, 'cat')
	rmdir_then_mkdir(cat_dir)

	ship_dir = os.path.join(aim_dir, 'ship')
	rmdir_then_mkdir(ship_dir)

	truck_dir = os.path.join(aim_dir, 'truck')
	rmdir_then_mkdir(truck_dir)

	deer_dir = os.path.join(aim_dir, 'deer')
	rmdir_then_mkdir(deer_dir)

	dog_dir = os.path.join(aim_dir, 'dog')
	rmdir_then_mkdir(dog_dir)

	airplane_dir = os.path.join(aim_dir, 'airplane')
	rmdir_then_mkdir(airplane_dir)

	# 初始化一个字典，通过控制其values 来控制总文件数
	label_count = {'frog': 0, 'automobile': 0, 'horse': 0, 'bird': 0, 'cat': 0,
				   'ship': 0, 'truck': 0, 'deer': 0, 'dog': 0, 'airplane': 0}

	for filename, label in zip(idx_label['id'], idx_label['label']):

		filename = str(filename) + '.png'
		# 判断每个类别的文件数是否满足数量条件：
		if label_count.get(label) < num_per_category:
			shutil.copy(os.path.join(train_dir, filename),
						os.path.join(aim_dir, label))

			# 相应类别的value +1，很神奇的一句话，哈哈哈
			label_count[label] = label_count.get(label) + 1

		# 判断如果所有文件夹的文件数之和为5000,则说明每个文件夹里有500个文件，可以终止

		elif sum(label_count.values()) >= num_per_category * 10:
			# print(label_count.values())
			break
		else:
			# print(label_count.values())
			continue



# 整理训练集
mv_file(idx_label_train, num_train_per_category, input_dir)
# 整理测试集
mv_file(idx_label_test, num_test_per_category, test_dir)