"""
import numpy as np

import mmap
import engine.PyPackerDetect.DetectPacker


class PackerExtractor(FeatureType):
    ''' Extracts doubt packer count '''

    name = 'Packer'
    dim = 1 + 1

    def __init__(self):
        super(FeatureType, self).__init__()

    def raw_features(self, bytez, lief_binary):
        filepath = ["./samples/normal/notepad.exe"]
        
        report = engine.PyPackerDetect.DetectPacker.CheckForPackersInMemory(bytez)
        
        return {
            'detections': report.detections,
            'suspicions': report.suspicions
        }

    def process_raw_features(self, raw_obj):
        return np.hstack([raw_obj['detections'], raw_obj['suspicions']]).astype(np.float32)



#directory check
filepath = ["./samples/normal/notepad.exe"]

reports = engine.PyPackerDetect.DetectPacker.CheckForPackers(filepath)

for file, report in reports.items():
    print(report.detections, report.suspicions)


fd = open(filepath[0], 'rb')
test = fd.read()

reports = engine.PyPackerDetect.DetectPacker.CheckForPackersInMemory(test)
print(reports.detections)

#reports = CheckForPackers(filepath)
#print(reports)
# config, default valuepefile 

"""

import re
import lief
import hashlib
import os
import numpy as np

LIEF_MAJOR, LIEF_MINOR, _ = lief.__version__.split('.')
LIEF_EXPORT_OBJECT = int(LIEF_MAJOR) > 0 or ( int(LIEF_MAJOR)==0 and int(LIEF_MINOR) >= 10 )


class FeatureType(object):
    ''' Base class from which each feature type may inherit '''

    name = ''
    dim = 0

    def __repr__(self):
        return '{}({})'.format(self.name, self.dim)

    def raw_features(self, bytez, lief_binary):
        ''' Generate a JSON-able representation of the file '''
        raise (NotImplementedError)

    def process_raw_features(self, raw_obj):
        ''' Generate a feature vector from the raw features '''
        raise (NotImplementedError)

    def feature_vector(self, bytez, lief_binary):
        ''' Directly calculate the feature vector from the sample itself. This should only be implemented differently
        if there are significant speedups to be gained from combining the two functions. '''
        return self.process_raw_features(self.raw_features(bytez, lief_binary))

class DataDirectories(FeatureType):
    ''' Extracts size and virtual address of the first 15 data directories '''

    name = 'datadirectories'
    dim = 15 * 2

    def __init__(self):
        super(FeatureType, self).__init__()
        self._name_order = [
            "EXPORT_TABLE", "IMPORT_TABLE", "RESOURCE_TABLE", "EXCEPTION_TABLE", "CERTIFICATE_TABLE",
            "BASE_RELOCATION_TABLE", "DEBUG", "ARCHITECTURE", "GLOBAL_PTR", "TLS_TABLE", "LOAD_CONFIG_TABLE",
            "BOUND_IMPORT", "IAT", "DELAY_IMPORT_DESCRIPTOR", "CLR_RUNTIME_HEADER"
        ]

    def raw_features(self, bytez, lief_binary):
        output = []
        if lief_binary is None:
            return output

        for data_directory in lief_binary.data_directories:
            output.append({
                "name": str(data_directory.type).replace("DATA_DIRECTORY.", ""),
                "size": data_directory.size,
                "virtual_address": data_directory.rva
            })
        return output

    def process_raw_features(self, raw_obj):
        features = np.zeros(2 * len(self._name_order), dtype=np.float32)
        for i in range(len(self._name_order)):
            if i < len(raw_obj):
                features[2 * i] = raw_obj[i]["size"]
                features[2 * i + 1] = raw_obj[i]["virtual_address"]
        return features

class StringExtractor(FeatureType):
    ''' Extracts strings from raw byte stream '''

    name = 'strings'
    dim = 1 + 1 + 1 + 96 + 1 + 1 + 1 + 1 + 1

    def __init__(self):
        super(FeatureType, self).__init__()
        # all consecutive runs of 0x20 - 0x7f that are 5+ characters
        self._allstrings = re.compile(b'[\x20-\x7f]{5,}')
        # occurances of the string 'C:\'.  Not actually extracting the path
        self._paths = re.compile(b'c:\\\\', re.IGNORECASE)
        # occurances of http:// or https://.  Not actually extracting the URLs
        self._urls = re.compile(b'https?://', re.IGNORECASE)
        # occurances of the string prefix HKEY_.  No actually extracting registry names
        self._registry = re.compile(b'HKEY_')
        # crude evidence of an MZ header (dropper?) somewhere in the byte stream
        self._mz = re.compile(b'MZ')

    def raw_features(self, bytez, lief_binary):
        allstrings = self._allstrings.findall(bytez)
        if allstrings:
            # statistics about strings:
            string_lengths = [len(s) for s in allstrings]
            avlength = sum(string_lengths) / len(string_lengths)
            # map printable characters 0x20 - 0x7f to an int array consisting of 0-95, inclusive
            as_shifted_string = [b - ord(b'\x20') for b in b''.join(allstrings)]
            c = np.bincount(as_shifted_string, minlength=96)  # histogram count
            # distribution of characters in printable strings
            csum = c.sum()
            p = c.astype(np.float32) / csum
            wh = np.where(c)[0]
            H = np.sum(-p[wh] * np.log2(p[wh]))  # entropy
        else:
            avlength = 0
            c = np.zeros((96,), dtype=np.float32)
            H = 0
            csum = 0

        return {
            'numstrings': len(allstrings),
            'avlength': avlength,
            'printabledist': c.tolist(),  # store non-normalized histogram
            'printables': int(csum),
            'entropy': float(H),
            'paths': len(self._paths.findall(bytez)),
            'urls': len(self._urls.findall(bytez)),
            'registry': len(self._registry.findall(bytez)),
            'MZ': len(self._mz.findall(bytez))
        }

    def process_raw_features(self, raw_obj):
        hist_divisor = float(raw_obj['printables']) if raw_obj['printables'] > 0 else 1.0
        return np.hstack([
            raw_obj['numstrings'], raw_obj['avlength'], raw_obj['printables'],
            np.asarray(raw_obj['printabledist']) / hist_divisor, raw_obj['entropy'], raw_obj['paths'], raw_obj['urls'],
            raw_obj['registry'], raw_obj['MZ']
        ]).astype(np.float32)

class PEFeatureExtractor(object):
    ''' Extract useful features from a PE file, and return as a vector of fixed size. '''

    def __init__(self, feature_version=2):
        self.features = [
            StringExtractor(),
            PackerExtractor()
        ]
        if feature_version == 1:
            if not lief.__version__.startswith("0.8.3"):
                print(f"WARNING: EMBER feature version 1 were computed using lief version 0.8.3-18d5b75")
                print(f"WARNING:   lief version {lief.__version__} found instead. There may be slight inconsistencies")
                print(f"WARNING:   in the feature calculations.")
        elif feature_version == 2:
            self.features.append(DataDirectories())
            if not lief.__version__.startswith("0.9.0"):
                print(f"WARNING: EMBER feature version 2 were computed using lief version 0.9.0-")
                print(f"WARNING:   lief version {lief.__version__} found instead. There may be slight inconsistencies")
                print(f"WARNING:   in the feature calculations.")
        else:
            raise Exception(f"EMBER feature version must be 1 or 2. Not {feature_version}")
        
        self.dim = sum([fe.dim for fe in self.features])

    def raw_features(self, bytez):
        lief_errors = (lief.bad_format, lief.bad_file, lief.pe_error, lief.parser_error, lief.read_out_of_bound,
                       RuntimeError)
        try:
            lief_binary = lief.PE.parse(list(bytez))
        except lief_errors as e:
            print("lief error: ", str(e))
            lief_binary = None
        except Exception:  # everything else (KeyboardInterrupt, SystemExit, ValueError):
            raise

        features = {"sha256": hashlib.sha256(bytez).hexdigest()}
        features.update({fe.name: fe.raw_features(bytez, lief_binary) for fe in self.features})
        for k, v in features.items():
            print(k, v)
        
        return features

    def process_raw_features(self, raw_obj):
        feature_vectors = [fe.process_raw_features(raw_obj[fe.name]) for fe in self.features]
        return np.hstack(feature_vectors).astype(np.float32)

    def feature_vector(self, bytez):
        return self.process_raw_features(self.raw_features(bytez))


def running_sample(file_data, feature_version=2):
    """
    Predict a PE file with an LightGBM model
    """
    extractor = PEFeatureExtractor(feature_version)
    features = np.array(extractor.feature_vector(file_data), dtype=np.float32)
    return features

import engine.PyPackerDetect.DetectPacker

class PackerExtractor(FeatureType):
    ''' Extracts doubt packer count '''

    name = 'Packer'
    dim = 1 + 1

    def __init__(self):
        super(FeatureType, self).__init__()

    def raw_features(self, bytez, lief_binary):
        report = engine.PyPackerDetect.DetectPacker.CheckForPackersInMemory(bytez)
        
        return {
            'detections': report.detections,
            'suspicions': report.suspicions
        }

    def process_raw_features(self, raw_obj):
        return np.hstack([raw_obj['detections'], raw_obj['suspicions']]).astype(np.float32)




if __name__ == "__main__":
    print("run main")
    path = "C:\\Users\\melpin5378\\Desktop\\python_test\\test_bin\\HxD.exe"
    if os.path.exists(path):
        print("read file")
        file_data = open(path, "rb").read()
        print("train start")
        final_features = running_sample(file_data)
        #print(final_features)
    else:
        print("{} does not exist".format(path))
            
            
