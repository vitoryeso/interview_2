# fixing the face dataset

## for the training of the network, simply some parameters were adjusted in the part of data increase, like increasing zoom range, disabling flip and rotations.

## other than that, a little digital image processing to rotate the images, data structures in python to extract and export the data, a little bit of  programming

### for use this dataset, you need to download the data and extract the data and run this codes

### "cifar10_cnn.py" train the network
### "data_pipe.py" have a util function to load the data for training
### "fix_data.py" get the trained model and fix the dataset 


## possible improvements not tested
### add some color changing in the data augmentation (HUE, gamma correction, etc)
### test other network backbones
### change the validation_data size



