################# fine-tune ######################
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.callbacks import TensorBoard
from keras.optimizers import RMSprop, Adam
from keras import backend as K


data_dir = '/home/jiang/Desktop/workspace/cifar-10/'
train_dir = data_dir + 'train_5000'
input_dir = data_dir + 'input_5000'
test_dir = data_dir + 'test_1000'
label_file = 'trainLabels.csv'

# create the base pre-trained model
base_model = InceptionResNetV2(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

# model.load_weights('./Inceptionv3_1.h5')

# 默认RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(optimizer=rmsprop,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# 数据增强参数
train_datagen = image.ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    fill_mode='nearest',
    horizontal_flip=True,
    vertical_flip=False,
    rescale=1./255)

test_datagen = image.ImageDataGenerator(rescale = 1./255)

# 指定数据增强文件夹
train_generator = train_datagen.flow_from_directory(
    input_dir,
    target_size = (299, 299),
    batch_size = 32,
    class_mode = 'categorical')

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size = (299, 299),
    batch_size = 32,
    class_mode = 'categorical')

history = model.fit_generator(
    train_generator,
    epochs = 100,
    steps_per_epoch = round(40000/32),
    validation_data = test_generator,
    validation_steps = round(1000/32),
    verbose=1,
    callbacks = [TensorBoard(log_dir = './log_InceptionResNetV2_3_rmsprop_0.001_40000')])

model.save_weights('InceptionResNetV2_3_rmsprop_0.001_40000.h5')
# model.save_weights(r'{}_fune_tune2.h5'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')))

# for i, layer in enumerate(base_model.layers):
#     print(i, layer.name)
