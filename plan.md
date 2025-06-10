# Bus & Number Detection Project



## Bus Detector

Data:
- COCO
- Singapore Buses

Model Architecture:
- YOLO11X

Steps:
1. Prepare script for COCO dataset download script
2. Find well labelled singapore data too
3. Finetune best model on the combined data up til 95% accuracy with just one class
4. Gather as much Singapore data as humanly possible
5. Auto-label buses in all the data available and crop it out
6. Process coco data to remove buses absolutely and save images
7. Improve bus crops by randomly halving certain images or making them super closeup
8. Perform stitching of singapore buses only onto images all around the world into the training script
9. Train on all that data on a x size model - go crazy on augmentations


## Number Detection

Data:
- Just singapore buses

Preparation Needed:
- Images with numbers labelled separately 
  - crop out and store numbers
  - keep the number less buses separate 
- Images with numbers labelled together 
  - Try cropping out based on ratio
  - Keep the number less buses separate
- Images with service board labeled 
  - Just remove the service board and keep the images
  - Can try using ocr to get number but not worth the work
- others
  - just crop out the buses, use ocr to remove any images of text
  - only need do this as last resort improvement

Augmentation needed:
- For the numbers try noising them
- Try covering different parts of the numbers while still being kinda visible

Steps:
- Crop out buses and stitch like crazy on all available images & backgrounds
- Train using YOLO11X
