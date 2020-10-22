import sys
import os
from PIL import Image
from tqdm import tqdm

def get_binary_data(binary_path: str) -> list:
    """
    악성 파일 데이터 수집
    :param binary_path: 바이너리 경로
    :return: 해당 악성 파일의 데이터
    """

    binary_data = []
    binary = open(binary_path, "rb")
    byte = binary.read(1)
    # 해당 바이너리를 1바이트씩 읽어 저장하고, 예외 발생시 통과
    while byte != b"":
        try:
            binary_data.append(ord(byte))
        except TypeError:
            pass

        byte = binary.read(1)
    binary.close()
    binary_data = binary_data[:1000] * 8 + binary_data

    return binary_data


def data_to_gray_image(binary_data: list, image_directory_path: str, binary_name: str):
    """
    바이너리 사이즈에 따라 width, height를 설정하여 저장
    :param binary_data: 악성 파일 데이터
    :param image_directory_path: 이미지 디렉토리 경로
    :param binary_name: 악성 파일 이름
    """

    data_size = len(binary_data)

    # 이미지 확장자
    extension = ".png"

    # 기본 값 설정
    base = 10240
    width_base = 32

    if data_size < base:
        width = width_base

    elif base <= data_size < base*3:
        width = width_base * 2

    elif base*3 <= data_size < base*6:
        width = width_base * 4

    elif base*6 <= data_size < base*10:
        width = width_base * 8

    elif base*10 <= data_size < base*20:
        width = width_base * 12

    elif base*20 <= data_size < base*50:
        width = width_base * 16

    elif base*50 <= data_size < base*100:
        width = width_base * 24

    else:
        width = width_base * 32

    height = int(data_size/width) + 1

    image = Image.new("L", (width, height))

    # 이미지 저장
    image.putdata(binary_data)
    image_name = binary_name.split(".")[0] + extension
    image.save(image_directory_path + "\\" + image_name)

    #print("[+] " + image_name + " saved")


def binary_to_image(binary_directory_path: str, image_directory_path: str):
    """
    바이너리 데이터를 수집하여 이미지로 변경
    :param binary_directory_path: 악성 파일 디렉토리 경로
    :param image_directory_path: 이미지 파일 디렉토리 경로
    """

    # 악성 디렉토리 내부의 파일 이름을 가져옴
    binary_list = os.listdir(binary_directory_path)

    # 악성 파일을 하나씩 이미지로 변환
    for idx in tqdm(range(len(binary_list))):
        # 악성 파일의 데이터
        binary_data = get_binary_data(binary_directory_path + "\\" + binary_list[idx])
        
        # 악성 파일 데이터를 이미지로 변환
        data_to_gray_image(binary_data, image_directory_path, binary_list[idx])


def main(binary_directory_path: str):
    """
    상위 디렉토리를 찾고, images 폴더를 생성
    :param binary_directory_path: 파일 폴더 경로
    """
    
    # 상위 디렉토리 경로
    upper_directory_path = os.path.dirname(binary_directory_path)
    
    # 이미지 디렉토리 경로
    image_directory_path = os.path.join(upper_directory_path, "images")

    # 이미지 디렉토리 존재 유무에 따른 생성
    os.makedirs(image_directory_path, exist_ok=True)

    # 파일을 이미지로 변환
    binary_to_image(binary_directory_path, image_directory_path)


if __name__ == '__main__':
    #test = "C:\\Users\\dodssockii\\Desktop\\test\\binary" # 파일경로 지정 시 사용
    #test = sys.argv[1] # 사용법 : python file.py [경로]
    inputpath = r"C:\Users\dlwlrma\Desktop\malware\266c05ab5a424c5d8621463d0bc6958a\train_set"
    output = r"C:\Users\dlwlrma\Desktop\capstone_git\capstone_design_project_2\sample\target_images"

    #main(test)
    #binary_to_image("../../dataset/malware", "../../dataset/images")
    binary_to_image(inputpath, output)