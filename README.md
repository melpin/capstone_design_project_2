
#lief package install method  
#not pip install lief < not running  
pip install --index-url  https://lief-project.github.io/packages lief  
https://github.com/lief-project/LIEF/releases  
lief 0.9.0 version install  
  
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
