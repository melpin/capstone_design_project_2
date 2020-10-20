# this file directory in /
# current directory in /test
# maybe this test file run, move dir

import csv
import operator
import os
from itertools import chain

import pefile
import mmap
from capstone import *

"""
class NGRAM_features(FeatureType):
    ''' Extracts doubt ngram count '''

    name = '4gram'
    dim = 100

    def __init__(self):
        super(FeatureType, self).__init__()
        self.imports = ""
        with open("./engine/ngram/4gram_database.csv", newline='') as db:
            headerdata = csv.reader(db, delimiter=',', quotechar='|')
            self.headers =  list(headerdata)[0]
            
            
    def gen_list_n_gram(self, num, asm_list):
        for i in range(0, len(asm_list), num):
            yield asm_list[i: i + num]

    def n_grams(self, num, asm_list):
        gram = dict()

        gen_list = self.gen_list_n_gram(num, asm_list)
        
        for lis in gen_list:
            lis = " ".join(lis)
            try:
                gram[lis] += 1
            except:
                gram[lis] = 1
                
        return gram

    def get_ngram_count(self, grams):
        patterns = list()

        for pat in self.headers:
            try:
                patterns.append(grams[pat])
            except:
                patterns.append(0)
                
        return patterns       

    def get_opcodes(self, bytez):

        asm = []
        
        pe = pefile.PE(data=bytez)

        ep = pe.OPTIONAL_HEADER.AddressOfEntryPoint
        end = pe.OPTIONAL_HEADER.SizeOfCode

        for section in pe.sections:
            addr = section.VirtualAddress
            size = section.Misc_VirtualSize

            if ep > addr and ep < (addr + size):
                ep = addr
                end = size

        data = pe.get_memory_mapped_image()[ep:ep + end]

        md = Cs(CS_ARCH_X86, CS_MODE_32)
        md.detail = False

        for insn in md.disasm(data, 0x401000):
            asm.append(insn.mnemonic)
        return asm

    def raw_features(self, bytez, lief_binary):
        byte_code = self.get_opcodes(bytez)
        grams = self.n_grams(4, byte_code)
        gram_count = self.get_ngram_count(grams)
        print(gram_count)
        
        return gram_count

    def process_raw_features(self, raw_obj):
        return np.hstack(raw_obj).astype(np.float32)
"""

class NGRAM_features:
    def __init__(self):
        self.imports = ""
        with open("./engine/ngram/4gram_database.csv", newline='') as db:
            headerdata = csv.reader(db, delimiter=',', quotechar='|')
            self.headers =  list(headerdata)[0]
        
    def gen_list_n_gram(self, num, asm_list):
        for i in range(0, len(asm_list), num):
            yield asm_list[i: i + num]

    def n_grams(self, num, asm_list):
        gram = dict()
        
        #gen_list = self.gen_list_n_gram(num, asm_list)
        gen_list = [asm_list[i:i+num] for i in range(0, len(asm_list), num)]
        
        for lis in gen_list:
            lis = " ".join(lis)
            try:
                gram[lis] += 1
            except:
                gram[lis] = 1

        return gram

    def get_ngram_count(self, grams):
        patterns = list()

        for pat in self.headers:
            try:
                patterns.append(grams[pat])
            except:
                patterns.append(0)

        return patterns

    def find_entry_point_section(self, pe, eop_rva):
        for section in pe.sections:
            if section.contains_rva(eop_rva):
                return section

        return None

    def get_opcodes(self, filepath):

        asm = []
        
        pe_data = open(filepath, 'rb').read()
        #fd = open(filepath, 'rb')
        #pe_data = mmap.mmap(bytez.fileno(), 0, access=mmap.ACCESS_READ)
        pe = pefile.PE(data=pe_data)
        
        #pe = pefile.PE(filepath)

        ep = pe.OPTIONAL_HEADER.AddressOfEntryPoint
        end = pe.OPTIONAL_HEADER.SizeOfCode

        for section in pe.sections:
            addr = section.VirtualAddress
            size = section.Misc_VirtualSize

            if ep > addr and ep < (addr + size):
                # print(section.Name)
                ep = addr
                end = size

        data = pe.get_memory_mapped_image()[ep:ep + end]

        md = Cs(CS_ARCH_X86, CS_MODE_32)
        md.detail = False

        for insn in md.disasm(data, 0x401000):
            # print("0x%x:\t%s\t%s" % (insn.address, insn.mnemonic, insn.op_str))
            # print(insn.mnemonic)
            asm.append(insn.mnemonic)
        return asm


def test():
    num_of_features = 100

    mal_path = './samples/malware/'
    path2 = "./samples/malware/0a0aca89c3064b40f78badadeb32c56b"
    ef = NGRAM_features()
    
    i = 0

    for file in os.listdir(mal_path):
        i += 1
        print("%d file processed (%s)," % (i, file), )
        file = mal_path + file
        byte_code = ef.get_opcodes(path2)
        grams = ef.n_grams(4, byte_code)
        #print(grams)
        break

    gram_count = ef.get_ngram_count(grams) # get counter, new list
    all_data = [file]
    all_data.extend(gram_count)
    print(all_data)
    print("done")

def header_save():
    num_of_features = 100

    mal_path = './samples/malware/'

    ef = NGRAM_features()
    i = 0

    # 파일 패턴 추출
    # 기존 패턴 존재 시 pattern_count +=1
    # 새로운 패턴 발견 시 pattern_count = 1
    # 관련 함수 : n_grams, gen_list_n_gram
    for file in os.listdir(mal_path):
        i += 1
        print("%d file processed (%s)," % (i, file), )
        file = mal_path + file
        byte_code = ef.get_opcodes(file)
        grams = ef.n_grams(4, byte_code)
        #print("grams : ",grams)
        #print("%d patterns extracted" % (len(grams)))

    #파일 패턴 빈도수를 기준으로 정렬,
    #모든 패턴을 사용하면 출력 파일 크기가 너무 커지기 때문에 필요에 따라 조정.

    # 현재 n = 500 으로 설정되어 있고, 이 부분이 헤더로,  (line 121. num_of_features = 500)
    # 헤더  { [file_name], [file_hash(md5)], [patterns_1]- - - - [patterns_n], [class] }
    sorted_x = sorted(grams.items(), key=operator.itemgetter(1), reverse=True)
    print("[*] Using %s grams as features" % num_of_features)
    features = sorted_x[0:num_of_features] # slice gram count
    headers = list(chain.from_iterable(zip(*features)))[0:num_of_features]
    print(headers)
    #header ranking
    filepath = "./engine/ngram/4gram_database.csv"
    fd = open(filepath, "w")
    for i in headers:
        fd.write(i+",")
    fd.close()
    
    csv_file = open(filepath, "w")
    writer = csv.writer(csv_file, delimiter=',')
    writer.writerow(headers)
    csv_file.close()
    #ranking header save routine



"""
add this routine

#write database
data = ['test', 'move move move', '234', 'ok']

filepath = "/engine/ngram/4gram_database.csv"
    fd = open(filepath, "w")
    for i in headers:
        fd.write(i+",")
    fd.close()

#read init database
filepath = "/engine/ngram/4gram_database.csv"
my_file = open(filepath, "r")
header = my_file.read().split(",")
print(header)
my_file.close()
"""


if __name__ == '__main__':
    #header_save()
    test()