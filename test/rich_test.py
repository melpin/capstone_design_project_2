# this file directory in /
# current directory in /test
# maybe this test file run, move dir

import hashlib
import os
import traceback
from itertools import chain
import numpy as np

# imports for rich
from engine.richheader import richlibrary


class RichHeader_features:
    name = 'richheader'
    dim = 271
    
    def richs(self, rich_g):
        rich = dict()
        for i in range(1, self.dim):
            rich[i] = 0;

        if(rich_g['error'] < 0):
            rich['error'] = rich_g['error']
            return rich

        cmpids = rich_g['cmpids']

        for lis in cmpids:
            rich[lis['pid']] = lis['cnt']
        print("rich ",rich)
        return rich

    def get_rich_data(self, file):
        filedata = open(file, "rb").read()
        rich_parser = richlibrary.RichLibrary(bytez = filedata)
        error = 0
        rich_data = {'error': 0, 'cmpids': [{'mcv': 0, 'pid': 0, 'cnt': 0}], 'csum_calc': 0, 'csum_file': 0, 'offset': 0}
        try:
            rich_data = rich_parser.parse()
        except richlibrary.FileSizeError:
            error = -2
        except richlibrary.MZSignatureError:
            error = -3
        except richlibrary.MZPointerError:
            error = -4
        except richlibrary.PESignatureError:
            error = -5
        except richlibrary.RichSignatureError:
            error = -6
        except richlibrary.DanSSignatureError:
            error = -7
        except richlibrary.HeaderPaddingError:
            error = -8
        except richlibrary.RichLengthError:
            error = -9
        except Exception as e:
            print(traceback.format_exc(e))
        rich_data['error'] = error
        return rich_data

    def debug_data_print(self, data):
        print("data")
        for a in data:
            print(a, end = ", ")
        print()


def main():
    mal_path = './samples/malware/'

    #print('[*] Extracting richs patterns from files')

    ef = RichHeader_features()
    i = 0

    print(os.listdir(mal_path))
    for file in os.listdir(mal_path):
        i += 1
        #print("%d file processd (%s), " % (i, file))
        file = mal_path + file
        rich_g = ef.get_rich_parser(file)

        richs = ef.richs(rich_g)
        #print("%d rich_pids extracted" % (len(richs)))

    headers = list(chain.from_iterable(zip(richs)))

    i = 0
    for file in os.listdir(mal_path):
        i += 1
        print("%d file processed (%s), " % (i, file))
        filepath = mal_path + file
        rich_c = ef.get_rich_parser(filepath)
        print("debug")
        richs = ef.richs(rich_c)
        
        #리치카운트에서 list 반환 이후 작업
        
        print(richs)
        all_data = [file]
        all_data.extend(richs)
        #ef.debug_data_print(all_data)
        
        #counts = np.array(all_data[1:], dtype=np.float32)
        #print(all_data)
    #return counts
    
        
if __name__ == '__main__':
    main()
