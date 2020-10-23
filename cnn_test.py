from engine.image_cnn import get_image_modeling_v2

if __name__ == '__main__':
    #modeldir = './sample/aimodel/'
    path = "./sample/target_images"
    label_path = "./sample/label.csv"
    train_model_path = "./sample/aimodel"
    
    testpath = "./dataset/malware"
    testlabel = "./dataset/testlabel.csv"
    testmodel_path = "./dataset/aimodel"
    
    #cn = get_image_modeling_v2.CNN_tensor(testpath, testlabel,testmodel_path)
    #cn.load_images()
    #cn.train()
    #test
    
    cn = get_image_modeling_v2.CNN_tensor(path, label_path, train_model_path)
    cn.load_images()
    cn.train()
