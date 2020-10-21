import tqdm
import features
import extractfeature
#import trainer
#import predictor
import utility

def main():

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

	trainsetdir = './sample/train_set'
	trainsetlabelpath = './sample/label.csv'
	trainsetfeaturepath = './sample/features.jsonl'

	testdir = "./dataset/malware"
	testlabel = "./dataset/testlabel.csv"
	testresult = "./dataset/features.jsonl"

	import time

	start = time.strftime('%m-%d, %H:%M:%S', time.localtime(time.time()))
    
	#extractor = extractfeature.Extractor(trainsetdir, trainsetlabelpath, trainsetfeaturepath, r)
	#extractor.run()

	extractor = extractfeature.Extractor(testdir, testlabel, testresult, r)
	extractor.run()
	#test extract

	end = time.strftime('%m-%d, %H:%M:%S', time.localtime(time.time()))

	import email_util

	subject = "capstone2 debug info"
	message = "process done\n"
	message += "runtime check\n"
	message += "start time : "+start + "\n"
	message += "end time : "+end + "\n"
	#email_util.debugmail(subject, message)
	#email alram routine

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

if __name__ == '__main__':
    main()
