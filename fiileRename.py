import os

path = './DataSet/train_2/'

# 获取该目录下所有文件，存入列表中
f = os.listdir(path)

n = 0
for i in f:
    # 设置旧文件名（就是路径+文件名）
    oldname = path + f[n]

    # 设置新文件名
    newname = path + str(n + 4654 + 1) + '.jpg'     # 4654为上一个数据集的末尾文件编号

    # 用os模块中的rename方法对文件改名
    os.rename(oldname, newname)
    print(oldname, '======>', newname)

    n += 1
