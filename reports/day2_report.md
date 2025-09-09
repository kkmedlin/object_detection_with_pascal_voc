##Pascal VOC Project – Development Reflection Questions
# Day 2 – Preprocessing & DataLoader

1. How did you setup the PyTorch Dataset? Did you use VOCDetection or a custom Dataset class? Why?

I used torchvision.datasets.VOCDetection to load the JPG images and XML annotation files. No need to write a custom Dataset class at this time.

2. How did you handle images of different sizes in the DataLoader?

I created a custom collate_fn for the DataLoader. By default, PyTorch's DataLoader tries to stack all samples in a batch, which doesn't work with tensors of different sizes. Instead of trying to stack tensors of different shapes, my custom collate_fn keeps images in a list. 

3. Why did you need a custom collate_fn for object detection?

In addition to havine images of different sizes, images have different numbers of annotations. a custom collate_fn is needed to i. keep image tensors as a list, and ii. group an image and its target dictionary into a list. This avoids two errors due to trying to stack different sized tensors -- one from different sizes images; another from differing number of annotations per image.  

4. How did you apply preprocessing transforms (resize, normalize, convert to tensor)?

I converted PIL Images to Torch Tensors, normalized pizel values, and applied random horizontal flips that also flipped bounding boxes. This ensures consistency for the model and adds diversity to the training set. 

5. What data augmentations did you consider? Why are they important?

Considered random horizontal flips and random color jitters.
They are important for adding robustness/diversity to a training set. They help decrease the likelihood of over-fitting.

6. How did you confirm that train/val splits were correct?

Pascal VOC 2012 already split into training and validations sets. I compared the number of training and validation images to how many Kaggle says there should be.

7. How did you sanity-check the DataLoader? What did you look for in batches?

I looped through batches of images and visualized the first image of the batch

8. What errors did you encounter when stacking tensors of different sizes? How did you resolve them?

Initially, I got a tensor stacking error due to images in Pascal VOC being different sizes. I resolved it by defining a custom collate_fn that returns lists of tensors intead of trying to stack image tensors into a single tensor.

9. How does preprocessing and batch preparation affect training stability and model performance?

It's critical. Consistent preprocessing stabilizes training. Proper batching avoids runtime errors. Data augmentation in preprocessing improves model performance by making the model more robust to over-fitting.

10. If you were to extend this project to YOLO or Faster-RCNN, what additional preprocessing steps or dataset considerations would you implement?

For YOLO, I'd need to resize all the images to a fixed input size (e.g., 640x640) and normalize bounding box coordinates to be relative to total width and height of the box. For Faster R-CNN, I'd keep variable image sizes but ensure annotations are in the expected dictionary format. In both cases, I'd expand augmentation (random crops, scale jittering) to improve robustness.