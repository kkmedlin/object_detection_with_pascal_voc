##Pascal VOC Project – Interview Questions
#Day 1 – Setup & Exploration

1. What is the Pascal VOC 2012 dataset? How is it structured (images, annotations, train/val splits)?

Pascal VOC 2012 dataset is made up of 17,125 jpeg images. It is a widely-used bench-mark dataset in computer vision for training and evaluating models on tasks like classificaiton, object detection, segmentation.

2. Why did you choose Pascal VOC 2012 instead of 2007?

I choose Pascal VOC 2012 instead of 2007 as the older version was no longer readily available from Kaggle.

3. How did you handle downloading and storing the dataset on Windows? Any challenges?

There were some initial challenges as I missed the step of having my .gitignore document setup to ignore data files. Therefore, when I 

4. How did you verify the dataset was complete and correctly extracted?

5. What tools and libraries did you use for initial exploration?

6. How did you visualize images and bounding boxes?

7. What did you learn about the variability in image sizes, aspect ratios, and bounding box formats?

8. How could dataset inspection influence model training or preprocessing choices?

9. If an image or annotation were missing or corrupted, how would you handle it?

10. What did you learn about reproducibility and project organization from this step?

Day 2 – Preprocessing & DataLoader

1. How did you implement the PyTorch Dataset? Did you use VOCDetection or a custom Dataset class? Why?

2. How did you handle images of different sizes in the DataLoader?

3. Why did you need a custom collate_fn for object detection?

4. How did you apply preprocessing transforms (resize, normalize, convert to tensor)?

5. What data augmentations did you consider? Why are they important?

6. How did you confirm that train/val splits were correct?

7. How did you sanity-check the DataLoader? What did you look for in batches?

What errors did you encounter when stacking tensors of different sizes? How did you resolve them?

How does preprocessing and batch preparation affect training stability and model performance?

If you were to extend this project to YOLO or Faster-RCNN, what additional preprocessing steps or dataset considerations would you implement?