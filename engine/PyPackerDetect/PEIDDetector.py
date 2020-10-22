import sys
import os

PyPackerDetecter_path = os.path.abspath("./engine/PyPackerDetect")
import peutils
from Utils import *
from PackerDetector import *

class PEIDDetector(PackerDetector):
    def __init__(self, config):
        super().__init__(config)
        
        try:
            if (self.config["UseLargePEIDDatabase"]):
                self.signatures = peutils.SignatureDatabase('deps/peid/signatures_long.txt')
            else:
                self.signatures = peutils.SignatureDatabase('deps/peid/signatures_short.txt')
        except:
            if (self.config["UseLargePEIDDatabase"]):
                self.signatures = peutils.SignatureDatabase(PyPackerDetecter_path+'/deps/peid/signatures_long.txt')
            else:
                self.signatures = peutils.SignatureDatabase(PyPackerDetecter_path+'/deps/peid/signatures_short.txt')

    def Run(self, pe, report):
        if (not self.config["CheckForPEIDSignatures"]):
            return

        matches = self.signatures.match_all(pe, ep_only=self.config["OnlyPEIDEntryPointSignatures"])
        if (not matches):
            return
        for match in matches:
            report.IndicateDetection("Found PEID signature: %s" % match)
