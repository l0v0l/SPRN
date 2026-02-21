# Source code of the SPRN

## Description of dataset

**Input for the segmentation network**
The input of the model is a 3-channel image: 
1. Ultrasound images;
2. MAB ROI;
3. Empty channel.

**Structure of the dataset**：
Root_folder
|_train_slices (Input for training)
|_train_sp_masks (MSM for training)
|_train_plaque_masks (plaque masks for training)
|_val_slices (Input for validation)
|_val_sp_masks (MSM for validation)
|_val_plaque_masks (plaque masks for validation)
|_test_slices (Input for testing)
|_test_sp_masks (MSM for testing)
|_test_plaque_masks (plaque masks for testing)


## Training
There are two training stage. Please run stage1.py first to generate the base model. Next, you can run stage2.py to get the final model for testing. 