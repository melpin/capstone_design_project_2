# this file directory in /
# current directory in /test
# maybe this test file run, move dir

import csv
import operator
import os
from itertools import chain

import pefile
from capstone import *

class NGRAM_features:
    def __init__(self):
        self.imports = ""
        self.gram = dict()
        
    def gen_list_n_gram(self, num, asm_list):
        for i in range(0, len(asm_list), num):
            yield asm_list[i: i + num]
    
    def n_grams_header_learn(self, num, asm_list):
        gram = self.gram

        gen_list = self.gen_list_n_gram(num, asm_list)

        for lis in gen_list:
            lis = " ".join(lis)
            try:
                gram[lis] += 1
            except:
                gram[lis] = 1

        return gram

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

def header_save():
    num_of_features = 500
    mal_path = './dataset/malware/'
    ef = NGRAM_features()
    i = 0

    for file in os.listdir(mal_path):
        i += 1
        print("%d file processed (%s)," % (i, file), )
        file = mal_path + file
        byte_code = ef.get_opcodes(file)
        grams = ef.n_grams_header_learn(4, byte_code)
    
    sorted_x = sorted(grams.items(), key=operator.itemgetter(1), reverse=True)
    print("[*] Using %s grams as features" % num_of_features)
    features = sorted_x[0:num_of_features] # slice gram count
    headers = list(chain.from_iterable(zip(*features)))[0:num_of_features]
    #header ranking
    filepath = "./engine/ngram/4gram_database.csv"
    
    csv_file = open(filepath, "w")
    writer = csv.writer(csv_file, delimiter=',')
    writer.writerow(headers)
    csv_file.close()
    #ranking header save routine



if __name__ == '__main__':
    header_save()
    