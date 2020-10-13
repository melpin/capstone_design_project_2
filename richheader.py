#!/usr/bin/env python3
# 출처 : https://github.com/HolmesProcessing/RichHeader-Service_Collection
# 참조
# https://infocon.hackingand.coffee/Hacktivity/Hacktivity%202016/Presentations/George_Webster-and-Julian-Kirsch.pdf?__cf_chl_jschl_tk__=365c68d8b1d8892dd6ffcfeb05c4a6d0043ed0ad-1600313813-0-AUDiL9fOGEOL0XmW7bHoxdqrNTOBHBJkyG-r43P2z7ZAIFyns2VtkjY0bVRyeXKO0jljtMFkoQYAc7mjtPC2-NJRnq1VF0gZVTS_2BjpA4BXgP18TEzjAq-PZPfP310AU7iE55QKFpYIbiOaXBVghBA_4QDLfxg3HGuUQ3iZjVRCVou0qLz1hnBKyP_181uh78Wa6AoLYSWMT5GJGkegwIaOz632VnsW1ubNfAWGlZpfI6r6XmwRMK1ctz58gdYEoNrMfYHrYc2CNse9_DVEokIXPc4beAHxw6SOYpGjaXIyZ1aKBv8z2h6MlAtlGnsGk_3xcC8VDrLQ-aHxWrkeQCdWKk_LbRkn2LpaBaSaYx2apz1oaXIPl6fAepRONyL7Zw
# https://www.youtube.com/watch?v=ftZQGDujLgo
# https://www.sans.org/reading-room/whitepapers/reverseengineeringmalware/paper/39045
# https://www.virusbulletin.com/virusbulletin/2020/01/vb2019-paper-rich-headers-leveraging-mysterious-artifact-pe-format/#ref13

import csv
import hashlib
import operator
import os
import traceback
from itertools import chain

# imports for rich
import richlibrary


class RichHeader_features:
    def __init__(self, output):
        self.output = output
        self.rich = dict()
        self.imports = ""

    def richs(self, rich_g, ex_mode):
        if ex_mode == 1:            
            rich = self.rich
        elif ex_mode == 0:
            rich = dict()

        if(rich_g['error'] < 0):
            rich['error'] = rich_g['error']
            return rich

        cmpids = rich_g['cmpids']

        for lis in cmpids:
            try:
                rich[lis['pid']] += lis['cnt']
            except:
                rich[lis['pid']] = lis['cnt']
        return rich

    def get_rich_count(self, headers, rich, label):
        pids = list()

        for pid in headers:
            try:
                pids.append(rich[pid])
            except:
                pids.append(0)

        pids.append(label)

        return pids

    def get_rich_parser(selfself, file):

        rich_parser = richlibrary.RichLibrary(file)
        error = 0
        rich = {'error': 0, 'cmpids': [{'mcv': 0, 'pid': 0, 'cnt': 0}], 'csum_calc': 0, 'csum_file': 0, 'offset': 0}
        try:
            rich = rich_parser.parse()
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
        rich['error'] = error

        return rich

    def getMD5(self, filepath):
        with open(filepath, 'rb') as fh:
            m = hashlib.md5()
            while True:
                data = fh.read(8192)
                if not data:
                    break
                m.update(data)
            return m.hexdigest()

    def write_csv_header(self, headers):
        filepath = self.output
        HASH = ['filename', 'MD5']
        class_ = ['class']
        headers = HASH + headers + class_

        csv_file = open(filepath, "w+")
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(headers)
        csv_file.close()

    def write_csv_data(self, data):
        filepath = self.output
        csv_file = open(filepath, "a")
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(data)
        csv_file.close()


def main():
    mal_path = './samples/malware/'
    nor_path = './samples/normal/'
    output_file = "./richd.csv"

    print('[*] Extracting ngram patterns from files')

    ef = RichHeader_features(output_file)
    i = 0

    for file in os.listdir(mal_path):
        i += 1
        print("%d file processd (%s), " % (i, file))
        file = mal_path + file
        rich_g = ef.get_rich_parser(file)

        richs = ef.richs(rich_g, 1)
        print("%d rich_pids extracted" % (len(richs)))

    print("- Malware Completed")
    print(richs)

    for file in os.listdir(nor_path):
        i += 1
        print("%d file processd (%s), " % (i, file))
        # file = nor_path + file
        file = "C:\\WINDOWS\\system32\\notepad.exe"
        rich_g = ef.get_rich_parser(file)

        richs = ef.richs(rich_g, 1)
        print("%d rich_pids extracted" % (len(richs)))

    print("- Normal Complated")
    print(richs)
    print("[*] Total length of rich_pid list :", len(richs))
    num_of_features = len(richs)

    sorted_x = sorted(richs.items(), key=operator.itemgetter(1), reverse=True)
    print("[*] Using %s richs as features" % num_of_features)
    features = sorted_x[0:num_of_features]
    headers = list(chain.from_iterable(zip(*features)))[0:num_of_features]
    ef.write_csv_header(headers)
    print(features)
    print("#" * 80)

    i = 0

    for file in os.listdir(mal_path):
        i += 1
        print("%d file processed (%s), " % (i, file))
        filepath = mal_path + file
        rich_c = ef.get_rich_parser(filepath)
        richs = ef.richs(rich_c, 0)

        rich_count = ef.get_rich_count(headers, richs, 1)
        hash_ = ef.getMD5(filepath)
        all_data = [file, hash_]
        all_data.extend(rich_count)
        ef.write_csv_data(all_data)

    for file in os.listdir(nor_path):
        i += 1
        print("%d file processed (%s), " % (i, file))
        # filepath = nor_path + file
        filepath = "C:\\WINDOWS\\system32\\notepad.exe"
        rich_c = ef.get_rich_parser(filepath)
        richs = ef.richs(rich_c, 0)

        rich_count = ef.get_rich_count(headers, richs, 0)
        hash_ = ef.getMD5(filepath)
        all_data = [file, hash_]
        all_data.extend(rich_count)
        ef.write_csv_data(all_data)


if __name__ == '__main__':
    main()
