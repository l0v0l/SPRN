# Source code of the SPRN

## Description of dataset

**Input for the segmentation network**
The input of the model is a 3-channel image: 
1. Ultrasound images;
2. MAB ROI;
3. Empty channel.

**Structure of the dataset**：

```bash
Root_folder
├──train_slices
├──train_sp_masks
├──train_plaque_masks
├──val_slices
├──val_sp_masks
├──val_plaque_masks
├──test_slices
├──test_sp_masks
└──test_plaque_masks
```

Explanation:
* train/val/test_slices: inputs for training/validation/testing.
* train/val/test_sp_masks: MSMs for training/validation/testing.
* train/val/test_plaque_masks: plaque masks for training/validation/testing.


## Training
There are two training stage. Please run stage1.py first to generate the base model. Next, you can run stage2.py to get the final model for testing. 

## Inference
After stage 2 training, the output images are stored in the folder for evaluation. The default folder name is 'output'

Or you can use the standalone test.py to generate segmentation

## Evaluation

Please run evaluation.py to get the experimental results. Three metrics are included: DSC, average TPV difference and 95 percentile of Hausdorff distance. 
