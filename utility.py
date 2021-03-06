import os
import json
import jsonlines
import features
import hashlib

def checkNone(_str_value):
    """
    Check if string is None
    """
    if not _str_value:
        return 0
    if _str_value == 'None':
        return 0

    return 1

def directory_generator(datadir):
    """
    Os.listdir to iterator
    """
    for sample in os.listdir(datadir):
        #print(sample)
        yield sample


def readonelineFromjson(jsonlpath):
    """
    Return features in JSONL.
    """
    with jsonlines.open(jsonlpath) as reader:
        for obj in reader:
            del obj['label']
            del obj['sha256']
            del obj['appeared']
            
            return list(obj.keys())

def get_file_hash(bytez = None, path = None):
    if bytez == None:
        data = open(path, "rb").read()
    else:
        data = bytez
    
    return hashlib.sha256(data).hexdigest()


class FeatureType:
    def __init__(self):
        self.names = {
            'histogram': features.ByteHistogram(),
            'byteentropy': features.ByteEntropyHistogram(),
            'section': features.SectionInfo(),
            'imports': features.ImportsInfo(),
            'exports': features.ExportsInfo(),
            'general': features.GeneralFileInfo(),
            'header': features.HeaderFileInfo(),
            'strings': features.StringExtractor(),
            'datadirectories': features.DataDirectories(),
            'Packer': features.PackerExtractor(),
            'richheader': features.RichHeader_features(),
            '4gram': features.NGRAM_features()
            
        }

    def parsing(self, lists):
        """
        return feature object for extracting
        """
        featurelist = []

        for feature in lists:
            if feature in self.names:
                featurelist.append(self.names.get(feature))

        return featurelist
