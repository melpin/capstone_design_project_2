#!/usr/bin/env python3
#출처 : https://github.com/HolmesProcessing/RichHeader-Service_Collection
#참조
#https://infocon.hackingand.coffee/Hacktivity/Hacktivity%202016/Presentations/George_Webster-and-Julian-Kirsch.pdf?__cf_chl_jschl_tk__=365c68d8b1d8892dd6ffcfeb05c4a6d0043ed0ad-1600313813-0-AUDiL9fOGEOL0XmW7bHoxdqrNTOBHBJkyG-r43P2z7ZAIFyns2VtkjY0bVRyeXKO0jljtMFkoQYAc7mjtPC2-NJRnq1VF0gZVTS_2BjpA4BXgP18TEzjAq-PZPfP310AU7iE55QKFpYIbiOaXBVghBA_4QDLfxg3HGuUQ3iZjVRCVou0qLz1hnBKyP_181uh78Wa6AoLYSWMT5GJGkegwIaOz632VnsW1ubNfAWGlZpfI6r6XmwRMK1ctz58gdYEoNrMfYHrYc2CNse9_DVEokIXPc4beAHxw6SOYpGjaXIyZ1aKBv8z2h6MlAtlGnsGk_3xcC8VDrLQ-aHxWrkeQCdWKk_LbRkn2LpaBaSaYx2apz1oaXIPl6fAepRONyL7Zw
#https://www.youtube.com/watch?v=ftZQGDujLgo
#https://www.sans.org/reading-room/whitepapers/reverseengineeringmalware/paper/39045
#https://www.virusbulletin.com/virusbulletin/2020/01/vb2019-paper-rich-headers-leveraging-mysterious-artifact-pe-format/#ref13

import sys
import traceback

# imports for rich
import richlibrary


file = "C:\\WINDOWS\\system32\\notepad.exe"
sys.argv = ["richheader.py", file]


def RichHeader(objpath):
    return richlibrary.RichLibrary(objpath)

def main():
    if len(sys.argv) < 2:
        print("Usage: {} <pe-files>".format(sys.argv[0]))
        sys.exit(-1)
    for arg in sys.argv[1:]:
        error = 0
        rich_parser = RichHeader(arg)

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

        #if error < 0:
        #    print("\x1b[33m[-] " + richlibrary.err2str(error) + "\x1b[39m")
        #    sys.exit(error)
        #else:
        #    rich_parser.pprint_header(rich)    
        #print(rich)
        if rich['error'] == -6:                     # RichHeader Not Found
            return 0
        elif rich['csum_calc'] != rich['csum_file']:# Checksum Not Equal
            return -1
        else :                                      # etc
            return rich['cmpids']
#etc부분 고민좀 해봐야 할거같음..
#단순 PID 뿐만 아니라 Count까지 비교해서 악성여부를 판단하기 때문..

if __name__ == '__main__':
    main()
