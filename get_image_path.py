import os,sys
from PIL import Image

def getBinaryData(filename):
    binaryValues = []
    file = open(filename, "rb")
    data = file.read(1)         # byte by byte로 읽는다.
    
    while data != b"":           # = data.decode('ascii')
        try:
            binaryValues.append(ord(data))      # store value to array(아스키코드값으로 바꿔서)

        except TypeError:
            pass

        data = file.read(1) # get next byte value


    return binaryValues



def createGrayscaleImage(dataSet, outputfilename, width=0):
    
    if (width == 0):            #크기가 지정되지 않았을때?(= don't specified) 
        size = len(dataSet)

    if (size < 10240):                  # 사이즈 별로 너비 설정을 따로 해야함.
        width = 32
    elif(10240 <= size <= 10240*3):
        width = 64
    elif(10240*3 <= size <= 10240*6):
        width = 128
    elif(10240*6 <= size <= 10240*10):
        width = 256
    elif(10240*10 <= size <= 10240*20):
        width = 384
    elif(10240*20 <= size <= 10240*50):
        width = 512
    elif(10240*50 <= size <= 10240*100):
        width = 768
    else:
        width = 1024

    height = int(size/width)+1              # 왜 +1인지 정확히는 모르겠지만, 예상은 나머지에 따른 값 보정인듯하다.(아니면 offest..)

    image = Image.new('L', (width, height)) # 8-bit pixels, black and white

    image.putdata(dataSet)                  # 위의 이미지 픽셀 데이터를 복사함.

    imagename = outputfilename+".png"
    image.save(imagename)
    image.show()                              # 결과확인용 
    print(imagename + "change over~")


if __name__=="__main__":                # 사용법 : [binary_to_image].py [해당 이미지화 할 바이너리 파일 풀 경로]
    file_full_path = sys.argv[1]
    path = os.path.dirname(file_full_path) # 디렉토리 경로 저장
    base_name = os.path.splitext(os.path.basename(file_full_path))[0] # (리스트로) 확장자만 따로 분류함.
    outputFilename = os.path.join(path, base_name) # 디렉토리 경로명 연결.

    binaryData=getBinaryData(file_full_path)
    createGrayscaleImage(binaryData, outputFilename)


