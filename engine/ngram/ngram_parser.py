import pefile
from capstone import *


def get_opcode_data(bytez):
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

def n_grams(num, asm_list):
    gram = dict()

    gen_list = [asm_list[i:i+num] for i in range(0, len(asm_list), num)]
    for lis in gen_list:
        lis = " ".join(lis)
        try:
            gram[lis] += 1
        except:
            gram[lis] = 1
            
    return gram

def get_ngram_dict(headers, grams):
    #dict version
    patterns = dict()
    for pat in headers:
        try:
            patterns[pat] = grams[pat]
        except:
            patterns[pat] = 0
    
    return patterns