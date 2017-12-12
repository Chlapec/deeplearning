# coding=utf-8


# 这个文件其实就是针对voc数据及进行 数据预处理 使得可以 在网络中 去使用

import os
import cv2
import xml.etree.ElementTree as ET
import numpy as np
import json


def get_data(input_path):
	all_imgs = []

	classes_count = {}

	class_mapping = {}

	visualise = False  # 默认可视化关闭了，这个变量在后面有一个应用

	data_paths = os.path.join(input_path, 'Annotations')

	print('Parsing annotation files')

	annot_path = os.path.join(data_paths, 'train.json')  # 图像注释路径
	imgs_path = os.path.join(data_paths, 'TrainImages')  # 原始图片路径
	# imgsets_path_trainval = os.path.join(data_path, 'ImageSets', 'Main', 'trainval.txt')
	# imgsets_path_test = os.path.join(data_path, 'ImageSets', 'Main', 'test.txt')

	# trainval_files = []
	# test_files = []
	# try:
	# 	with open(
	# 			imgsets_path_trainval) as f:  # 将ImageSets/Main/trainval.txt（其实这里面标记了可以拿去训练，既作训练集的图片的名称）打开，其中的每一行加上.jpg存入trainval_files
	# 		for line in f:
	# 			trainval_files.append(line.strip() + '.jpg')
	# except Exception as e:
	# 	print(e)

	f = open(annot_path)  # 打开json文件
	annots = json.load(f)
	#annots = json_detail[0]['items'][0]['label_id']

	idx = 0  # 一个索引
	for annot in annots:  # Annotations中的每一个文件（对应一个图片的bounding box坐标）
		try:
			idx += 1

			# et = ET.parse(annot)  # ET就是xml.etree.ElementTree，实现了一个简单而有效的用户解析和创建XML数据的API。
			# element = et.getroot()  # 对每一张图片获取根（名称：人，飞机，自行车等）

			#element_objs = element.findall('object')  # 查找所有的object标签
			element_objs = annot['items']  #element_objs列表 里面存储所有的对象及其具体信息


			#element_filename = element.find('filename').text  # 获取filename标签下的文本信息，如 <filename>2007_000027.jpg</filename>
			element_filename = annot['image_id']  #element_filename 字符串 存储了照片名字

			img = cv2.imread(os.path.join(imgs_path, element_filename)) #读取图片为了获取图片的 长 宽
			sp = img.shape
			#element_width = int(element.find('size').find('width').text)  # 同上 但是强制类型转换了
			element_width = int(sp[1])
			#element_height = int(element.find('size').find('height').text)
			element_height = int(sp[0])

			if len(element_objs) > 0:  # >0说明确实有 对象
				annotation_data = {'filepath': os.path.join(imgs_path, element_filename), 'width': element_width,
								   # 一个annotation_data是一个图片中的各种数据的字典
								   'height': element_height,
								   'bboxes': []}  # 根据xml中获取的element_filename结合前面的imgs_path,可以在原始图像目录中定位一张图片

				# if element_filename in trainval_files:  # Annotation中的xml所标记的图片只有记录在trainval.txt中的才会设置这个标签，也只有有这个标签的后面才会拿去训练
				# 	annotation_data['imageset'] = 'trainval'
				# elif element_filename in test_files:
				# 	annotation_data['imageset'] = 'test'
				# else:
				# 	annotation_data['imageset'] = 'trainval'

			for element_obj in element_objs:
				class_name = str(element_obj['label_id'])  # 获取 对象 的name 比如：person、horse等
				if class_name not in classes_count:  # 用来统计当前的这类别对象有多少个实例
					classes_count[class_name] = 1  # 如果原来没有这类对象，在class_count中增加这个项，初始化为1
				else:
					classes_count[class_name] += 1  # 如果有的话+1

				if class_name not in class_mapping:  # 如果不在mapping中的情况
					class_mapping[class_name] = len(class_mapping)  # 给检索到的类别 进行排序 通过长度0,1,2,3,4...

				obj_bbox = element_obj['bbox']  # 获取boundingbox坐标 #妈的 傻逼CCF这里要记得转换成voc格式 最后在转换回去


				x1 = int(round(float(obj_bbox[0])))
				y1 = int(round(float(obj_bbox[1])))
				x2 = int(round(float(obj_bbox[2])))
				y2 = int(round(float(obj_bbox[3])))


				#difficulty = int(element_obj.find('difficult').text) == 1
				annotation_data['bboxes'].append(
					{'class': class_name, 'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2,
					 'difficult': 1})  # 在这里终于将最后的bboxes条目补充满了
			all_imgs.append(annotation_data)  # 在这里一幅图片的所有信息已经储存完毕，现在将这一组信息，既annotation_data放入all_imgs字典中

			# if visualise:  # 前面被设置为false
			# 	img = cv2.imread(annotation_data['filepath'])
			# 	for bbox in annotation_data['bboxes']:
			# 		cv2.rectangle(img, (bbox['x1'], bbox['y1']), (bbox[
			# 														  'x2'], bbox['y2']), (0, 0, 255))
			# 	cv2.imshow('img', img)
			# 	cv2.waitKey(0)

		except Exception as e:  # 如果前面的一系列都被设置为错误的，则这里就要抛出来异常。
			print(e)
			continue
	return all_imgs, classes_count, class_mapping  # 返回 1.所有的每一张图片的具体信息（坐标，名字等等）2.所有图片算在一起 每一类的实例的数目 3.每一个类的名字的长度person=6，horse=5
