
class Config:

    ## Dataset
    training_dir = "/home/mmv/Documents/3.datasets/openlogo/preproc/1/training/"
    testing_dir = "/home/mmv/Documents/3.datasets/openlogo/preproc/1/testing/"

    # Alexnet 224,224 or 300,300
    im_w = 224
    im_h = 224

    ## Model params
    pretrained = False

    train_batch_size = 32
    train_number_epochs = 200
    lrate = 0.0005

    ## Model save/load path
    best_model_path = "testmodel.pt"
    model_path = "testmodel_last.pt"

