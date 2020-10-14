#python -m pip install --upgrade pip 도 필요해서 설치함.
import re
import lief
import hashlib
import numpy as np
from sklearn.feature_extraction import FeatureHasher

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

class ByteHistogram(FeatureType):
    # Byte histogram (count + non-normalized) over the entire binary file 

    name = 'histogram'
    dim = 256

    def __init__(self):
        super(FeatureType, self).__init__()

    def raw_features(self, bytez, lief_binary):
        counts = np.bincount(np.frombuffer(bytez, dtype=np.uint8), minlength=256)
        return counts.tolist()

    def process_raw_features(self, raw_obj):
        counts = np.array(raw_obj, dtype=np.float32)
        sum = counts.sum()
        normalized = counts / sum
        return normalized

class ByteEntropyHistogram(FeatureType):

    ''' 2d byte/entropy histogram based loosely on (Saxe and Berlin, 2015).
    This roughly approximates the joint probability of byte value and local entropy.
    See Section 2.1.1 in https://arxiv.org/pdf/1508.03096.pdf for more info.
    '''

    name = 'byteentropy'
    dim = 256

    def __init__(self, step=1024, window=2048):
        super(FeatureType, self).__init__()
        self.window = window
        self.step = step

    def _entropy_bin_counts(self, block):
        # bin 당 16바이트이며, coarse 히스토그램(=정리가 되어있지 않은 히스토그램)
        c = np.bincount(block >> 4, minlength=16)  # 16-bin histogram
        p = c.astype(np.float32) / self.window
        wh = np.where(c)[0]
        H = np.sum(-p[wh] * np.log2(
            p[wh])) * 2  # * x2 b.c. we reduced information by half: 256 bins (8 bits) to 16 bins (4 bits) == 8비트를 4비트 줄였다.

        Hbin = int(H * 2)  # up to 16 bins (max entropy is 8 bits)
        if Hbin == 16:  # handle entropy = 8.0 bits(비트당 2개니깐.)
            Hbin = 15

        return Hbin, c

class SectionInfo(FeatureType):
    ''' Information about section names, sizes and entropy.  Uses hashing trick
    to summarize all this section info into a feature vector.
    '''

    name = 'section'
    dim = 5 + 50 + 50 + 50 + 50 + 50

    def __init__(self):
        super(FeatureType, self).__init__()

    @staticmethod
    def _properties(s):
        return [str(c).split('.')[-1] for c in s.characteristics_lists]

    def raw_features(self, bytez, lief_binary):
        if lief_binary is None:
            return {"entry": "", "sections": []}

        # properties of entry point, or if invalid, the first executable section(= 진입 점의 속성 또는 유효하지 않은 경우)
        try:
            entry_section = lief_binary.section_from_offset(lief_binary.entrypoint).name
        except lief.not_found:
            # bad entry point, let's find the first executable section
            entry_section = ""
            for s in lief_binary.sections:
                if lief.PE.SECTION_CHARACTERISTICS.MEM_EXECUTE in s.characteristics_lists:
                    entry_section = s.name
                    break

        raw_obj = {"entry": entry_section}
        raw_obj["sections"] = [{
            'name': s.name,
            'size': s.size,
            'entropy': s.entropy,
            'vsize': s.virtual_size,
            'props': self._properties(s)
        } for s in lief_binary.sections]
        return raw_obj

    def process_raw_features(self, raw_obj):
        sections = raw_obj['sections']
        general = [
            len(sections),  # total number of sections
            # number of sections with nonzero size
            sum(1 for s in sections if s['size'] == 0),
            # number of sections with an empty name
            sum(1 for s in sections if s['name'] == ""),
            # number of RX
            sum(1 for s in sections if 'MEM_READ' in s['props'] and 'MEM_EXECUTE' in s['props']),
            # number of W
            sum(1 for s in sections if 'MEM_WRITE' in s['props'])
        ]
        # gross characteristics of each section
        section_sizes = [(s['name'], s['size']) for s in sections]
        section_sizes_hashed = FeatureHasher(50, input_type="pair").transform([section_sizes]).toarray()[0]
        section_entropy = [(s['name'], s['entropy']) for s in sections]
        section_entropy_hashed = FeatureHasher(50, input_type="pair").transform([section_entropy]).toarray()[0]
        section_vsize = [(s['name'], s['vsize']) for s in sections]
        section_vsize_hashed = FeatureHasher(50, input_type="pair").transform([section_vsize]).toarray()[0]
        entry_name_hashed = FeatureHasher(50, input_type="string").transform([raw_obj['entry']]).toarray()[0]
        characteristics = [p for s in sections for p in s['props'] if s['name'] == raw_obj['entry']]
        characteristics_hashed = FeatureHasher(50, input_type="string").transform([characteristics]).toarray()[0]

        return np.hstack([
            general, section_sizes_hashed, section_entropy_hashed, section_vsize_hashed, entry_name_hashed,
            characteristics_hashed
        ]).astype(np.float32)