
#lief package install method  
#not pip install lief < not running  
pip install --index-url  https://lief-project.github.io/packages lief  
#https://github.com/lief-project/LIEF/releases  

malware detection with ai

module decryption


(entropy)
sh_entropy.py => use sys, math (따로 인스톨 할 필요 없음.)  
ai_entropy.py(ember entropy code) : 기본 내장 라이브러리 외에 설치가 필요한 것만 넣어 놓음.  
=> pip install requests(아마 전부 있을거임.)  
=> pip install lief  
=> pip install numpy  
=> pip install sklearn  
=> python -m pip install --upgrade pip 도 한번 해주세요~  

richheader.py   
sample directory의 파일들로부터 추출한 richheader count & pid를 richd.csv로 출력  
with richlibrary.py, prodids.py  

ngram.py   
sample directory의 파일들로부터 추출한 segment count & 4-gram opcodes를 4-gram.csv로 출력  
  
  
(image)  
get_image.py  
해당 코드는 아래와 같이 경로를 만들어주어야 실행 가능
'../samples/malware/'  
'../samples/normal/'  
'../images/malware/'  
'../images/normal/'  
  
get_image_modeling_v2.py  
tensorflow-cpu 최신버전에서 사용가능  
코드를 Tensorflow 1.x 버전에서 Tensorflow 2.x 버전으로 변경하고 싶으면 명령어 창에 아래와 같이 작성  
=> tf_upgrade_v2 --infile (origin_file.py) --outfile (after_file.py)
