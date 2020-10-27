import tqdm
import features
import extractfeature
import trainer
import predictor
import utility
import time
import email_util


def parsing(targetdir, labeldir, jsonldir):

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

	start = time.strftime('%m-%d, %H:%M:%S', time.localtime(time.time()))
	#extractor = extractfeature.Extractor(trainsetdir, trainsetlabelpath, trainsetfeaturepath, r)
	#extractor.run()

	extractor = extractfeature.Extractor(targetdir, labeldir, jsonldir, r)
	extractor.run()

	#extractor = extractfeature.Extractor(testdir, testlabel, testresult, r)
	#extractor.run()
	#test extract
	end = time.strftime('%m-%d, %H:%M:%S', time.localtime(time.time()))



	subject = "capstone2 debug info"
	message = "parsing done\n"
	message += "runtime check\n"
	message += "start time : "+start + "\n"
	message += "end time : "+end + "\n"
	#email_util.debugmail(subject, message)
	#email alram routine

def train():
    modeldir = './sample/aimodel/'
    trainsetfeaturepath = './sample/features.jsonl'
    
    start = time.strftime('%m-%d, %H:%M:%S', time.localtime(time.time()))
    train = trainer.Trainer(trainsetfeaturepath, modeldir)
    train.run()
    #train process

    end = time.strftime('%m-%d, %H:%M:%S', time.localtime(time.time()))

    #email alram routine
    subject = "capstone2 debug info"
    message = "model save done\n"
    message += "runtime check\n"
    message += "start time : "+ start + "\n"
    message += "end time : "+end + "\n"
    #email_util.debugmail(subject, message)
    #email alram routine

def predict(sampledir, outputpath):
    featurelist = './sample/features.jsonl' # parsed data feature input file
    features = utility.readonelineFromjson(featurelist)
    feature_parser = utility.FeatureType()
    featureobjs = feature_parser.parsing(features)

    lgbmodelpath = './sample/aimodel/GradientBoosted_model.txt'
    xgbmodelpath = './sample/aimodel/X_GradientBoosted_model.txt'
    rfmodelpath = './sample/aimodel/RandomForest_model.txt'
    
    start = time.strftime('%m-%d, %H:%M:%S', time.localtime(time.time()))
    
    #all sample testing
    
    predict = predictor.Predictor(sampledir, featureobjs, outputpath)
    predict.lgbmodel_load(lgbmodelpath)
    #predict.xgbmodel_load(xgbmodelpath)
    #predict.rfmodel_load(rfmodelpath)
    
    predict.run()
    #predict process
    
    end = time.strftime('%m-%d, %H:%M:%S', time.localtime(time.time()))
    
    #email alram routine
    subject = "capstone2 debug info"
    message = "predict csv save done\n"
    message += "runtime check\n"
    message += "start time : "+start + "\n"
    message += "end time : "+end + "\n"
    email_util.debugmail(subject, message)
    #email alram routine


if __name__ == '__main__':
    #malware10000_dir = r"C:\Users\dlwlrma\Desktop\malware\266c05ab5a424c5d8621463d0bc6958a\train_set"
    sampledir = './sample/testset/' # data input folder
    outputpath = './sample/result.csv'
    
    targetdir = r"E:\target\1st_problem_set"
    targetoutputdir = r"E:\target\result.csv"
    targetjsonldir = r"E:\target\features.jsonl"
    
    #parsing(targetdir, "testlabel", targetjsonldir)
    train()
    #predict(targetdir, outputpath)
    
    
    #hash = utility.get_file_hash(bytez = None, path = dd)
    #get file hash test
    
    