# Malware detection with AI (LGBM)  

myai.py  
main app으로 여기서 extracture, trainer, predictor를 불러들여 실행  
features 에 기존 ember에서 ngram, rich header, pypacker detector 여부등을 추가  
lgbm 개선을 위해 optuna 적용 


richlibrary.py  
byte data 받을수 있도록 수정함

richheader.py   
sample directory의 파일들로부터 추출한 richheader count & pid를 richd.csv로 출력  
with richlibrary.py, prodids.py  

ngram.py   

header database file  
./engine/ngram/4gram_database.csv  

sample directory의 파일들로부터 추출한 segment count & 4-gram opcodes를 4-gram.csv로 출력  

PyPackerDetect-master  

peutils 408 line #  encode cp-949 > utf8 modify  
  
sig_f =open(filename, 'rt', encoding='UTF8')  
sig_f = open( filename, 'rt' )  
  
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
get_image.py와 같이 경로를 만들어 주어야 파일을 찾을 수 있도록 만듦  
아래와 같이 디렉토리 생성  

<pre>
<code>
Project
  ㄴㅡㅡ samples  
    ㄴㅡㅡ malwares  
    ㄴㅡㅡ normal  
  ㄴㅡㅡ images  
    ㄴㅡㅡ malwares  
    ㄴㅡㅡ normal  
  ㄴㅡㅡ engine  
    ㄴㅡㅡ code (code 파일)  
</code>
</pre>

ref https://github.com/lief-project/LIEF/releases  
ref https://github.com/cylance/PyPackerDetect  
ref https://github.com/elastic/ember
ref https://github.com/optuna/optuna/blob/master/examples/lightgbm_simple.py
