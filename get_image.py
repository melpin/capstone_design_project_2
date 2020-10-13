'''
pip install imageio
pip install numpy
pip install PIL
'''

import numpy as np
import os, array
import imageio
import glob
from PIL import Image



class IMAGE_feature():

    def __init__(self, in_path, out_path):

        self.in_path = in_path
        self.out_path = out_path


    def get_image(self, path, file):

            filename = path + file

            f = open(filename,'rb')
            ln = os.path.getsize(filename) # 파일 길이(바이트 단위)

            width = int(ln**0.5) # 파일 길이의 제곱근을 구함(정사각형 모양 이미지 위해)
            rem = ln % width 

            a = array.array("B") # unit8 배열
            a.fromfile(f,ln-rem) # 파일의 바이너리로 배열을 구성
            f.close() 

            g = np.reshape(a, (int(len(a)/width), width)) # 배열의 모양을 정사각형으로 재조정
            g = np.uint8(g)

            fpng = self.out_path + file + ".png"
            imageio.imsave(fpng, g) # 정사각형 모양의 배열을 그대로 이미지 형태로 저장

            outfile = self.out_path + file + "_thumb.png"
            print(outfile)
            size = 256, 256

            if fpng != outfile:
                im = Image.open(fpng)
                im.thumbnail(size, Image.ANTIALIAS) # 이미지를 256X256 크기의 썸네일로 생성
                im.save(outfile, "PNG")

    def get_all(self):
        path = self.in_path

        for file in os.listdir(path): 
            self.get_image(path, file)



def main():

    mal_path = '../samples/malware/'
    nor_path = '../samples/normal/'

    mal_out_path = '../images/malware/'
    nor_out_path = '../images/normal/'

    im1 = IMAGE_feature(mal_path, mal_out_path)
    im1.get_all()

    im2 = IMAGE_feature(nor_path, nor_out_path)
    im2.get_all()


if __name__ == '__main__':
    main()