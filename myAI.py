import tqdm
import features
import extractfeature
import trainer
import predictor
import utility

r = []
r.append(features.ByteHistogram())
r.append(features.ByteEntropyHistogram())
r.append(features.SectionInfo())
r.append(features.ImportsInfo())
r.append(features.ExportsInfo())
r.append(features.GeneralFileInfo())
r.append(features.HeaderFileInfo())
r.append(features.StringExtractor())
r.append(features.DataDirectories())
r.append(features.PackerExtractor())
r.append(features.RichHeader_features())
r.append(features.NGRAM_features())

trainsetdir = './sample/trainset/'
trainsetlabelpath = './sample/label.csv'
trainsetfeaturepath = './sample/features.jsonl'

import time

start = time.time()
extractor = extractfeature.Extractor(trainsetdir, trainsetlabelpath, trainsetfeaturepath, r)
extractor.run()

end = time.time()

print("done")
with open("result.txt", "w") as f:
	f.write("result runtime : ", end - start, " sec")
#extracte process

"""
modeldir = '/data/myAI/dataset/aimodel/'
train = trainer.Trainer(trainsetfeaturepath, modeldir)
train.run()
#train process
"""

"""
featurelist = '/data/myAI/dataset/features.jsonl'
features = utility.readonelineFromjson(featurelist)
feature_parser = utility.FeatureType()
featureobjs = feature_parser.parsing(features)

modelpath = '/data/myAI/dataset/aimodel/GradientBoosted_model.txt'
testdir = '/data/myAI/dataset/testset/'
outputpath = '/data/myAI/dataset/result.csv'
predict = predictor.Predictor(modelpath, testdir, featureobjs, outputpath)
predict.run()
#predict process
"""