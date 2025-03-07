'''
Author      : now more
Contact     : lin.honghui@qq.com
LastEditors: Please set LastEditors
LastEditTime: 2020-03-01 00:43:51
Description : 
'''
import zipfile
import os

def zipDir(dirpath,outFullName):
    """
    压缩指定文件夹
    :param dirpath: 目标文件夹路径
    :param outFullName: 压缩文件保存路径+xxxx.zip
    :return: 无
    """
    zip = zipfile.ZipFile(outFullName,"w",zipfile.ZIP_DEFLATED)
    for path,dirnames,filenames in os.walk(dirpath):
        # 去掉目标跟路径，只对目标文件夹下边的文件及文件夹进行压缩
        fpath = path.replace(dirpath,'')

        for filename in filenames:
            zip.write(os.path.join(path,filename),os.path.join(fpath,filename))
    zip.close()

if __name__ == "__main__":
    dir_path = r"/home/LinHonghui/Project/DrivenData_2020_SegBulid/exp/zip_test"
    outFullName = r"/home/LinHonghui/Project/DrivenData_2020_SegBulid/exp/zip_test.zip"
    zipDir(dir_path,outFullName)