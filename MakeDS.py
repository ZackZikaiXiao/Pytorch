import os
DIRECTORY= "./DataSet/train_2"#这里是自己子文件夹的图片的位置，train_1到train_n
f = open('./DataSet/train_2.txt','w') #txt文件位置train_1到train_n
files=os.listdir(DIRECTORY)
for file in files:
		f.writelines(file + " " + '2')
		f.write('\n')
f.close

# 详细请见：https://blog.csdn.net/Mr_FengT/article/details/90814237