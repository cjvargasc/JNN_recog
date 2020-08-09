
class Config:

    ## OpenLogo
    # training_dir = "openlogo training dataset"
    # testing_dir = "openlogo testing dataset"
    ## MiniImagenet
    training_dir = "miniimagenet training dataset"
    testing_dir = "miniimagenet testing dataset"

    # Alexnet 224,224 or 300,300
    im_w = 224
    im_h = 224

    ## Model params
    continue_training = False
    pretrained = False

    train_batch_size = 64
    train_number_epochs = 900
    lrate = 0.005

    ## Model save/load path
    best_model_path = "testmodel"
    model_path = "testmodel_last"

