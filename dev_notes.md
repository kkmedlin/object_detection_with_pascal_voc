##Pascal VOC Project – Interview Questions
# Day 1 – Setup & Exploration

1. What is the Pascal VOC 2012 dataset? How is it structured (images, annotations, train/val splits)?

Pascal VOC 2012 dataset is made up of 17,125 jpeg images. It is a widely-used bench-mark dataset in computer vision for training and evaluating models on tasks like classificaiton, object detection, segmentation.

2. Why did you choose Pascal VOC 2012 instead of 2007?

I choose Pascal VOC 2012 instead of 2007 as the older version was no longer readily available from Kaggle.

3. How did you handle downloading and storing the dataset on Windows? Any challenges?

There were some initial challenges as I missed the step of having my .gitignore document ignore data files. I was unable to synch with git due to there being too much data. So I started over. Now it works great, and I shall remember the importance of a well-crafted .gitignore file!

4. How did you verify the dataset was complete and correctly extracted?

I included print statements along the way confirming "Download complete!" and "Extraction complete!". Furthermore, I confirmed the number or .jpg images (17125) and 'xml annotation files (17125). Lastly, I displayed a sample image as well as a sample image with bounding box.

5. What tools and libraries did you use for initial exploration?

a. os - file and directory manipulation
b. subprocess -- allows local Python script to wait for the external commands (kaggle, datasets, dowload) to complete
c. matplotlib.pyplot -- plotting images
d. matplotlib.patches -- drawing rectangles
e. Image from PIL -- transforming .jpeg to plottable pixels
f. xml.etree.ElementTree -- parsing annotation into tree object
g. Counter from collections -- count numer of types of labels
   
6. How did you visualize images and bounding boxes?

To visualize the images:
I used the Image class from Python's Pillow library to open the .jpg image files and create an Image object. This step reads the .jpg files' binary data and decodes it into a format that represents the image's pixels.
Then I used Matplotlib's pyplot module to display the image data using .imshow(img) and plt.show()

To visualize the bounding boxes:
a. create Image object from Image class in PIL for a single image .jpeg file
b. create an ElementTree object using ET.parse(ann_path) for a single annotation .xml file -- tree is made up of different labels for single image
c. for each node in a tree ("for obj in root.findall("object")"), find its label and bounding boxes using obj.find
d. draw bounding boxes using patches.Rectangle


7. What did you learn about the variability in image sizes, aspect ratios, and bounding box formats?

a. VOC imgages come in a variety of image sizes
b. the variety will need to be addresses as many deep learning models (ResNet, YOLO, Faster R-CNN) requre consistent input sizes
c. Bounding boxes are stored as absolute pixel coordinates (xmin/max; ymin/max). These will need to be updated when resizing images.

8. How could dataset inspection influence model training or preprocessing choices?

a. when images have very different height and widths, they will need padding ("letterboxing") to avoid distortion 
b. when bounding boxes are very small compared to image size, data augmentation (cropping/zooming) may be required
c. distribution of class labels alerts you to possible class imbalanced problems
d. inspection shows whether annotations are consistent and, therefore, whether evaluation metrics will be valid

9. If an image or annotation were missing or corrupted, how would you handle it?

a. add an error handling step into a custom Dataset class
b. if file is corrupted, skip it (don't crash training)
c. if annotation is missing, either skip it or mark it as unlabeled
d. logging missing/corrupted data is important for reproducibility and later debugging

10. What did you learn about reproducibility and project organization from this step?

a. clear project structure (data/, notebooks/, scripts/, reports/) helps keep raw data, experiments and results separate
b. .gitignore ensures large files (like VOC images) don't clutter repo
c. keeping a Day1_setup notebook ensures the ability to confirm an intact dataset in the future

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

Initially, I got a tensor stacking error due to images in Pascal VOC being different sizes. I resolved it by defining a custom collate_fn that returns lists of tensorts intead of trying to stack image tensors into a single tensor.

9. How does preprocessing and batch preparation affect training stability and model performance?

It's critical. Consistent preprocessing stabilizes training. Proper batching avoids runtime errors. Data augmentation in preprocessing improves model performance by making the model more robust to over-fitting.

10. If you were to extend this project to YOLO or Faster-RCNN, what additional preprocessing steps or dataset considerations would you implement?

For YOLO, I'd need to resize all the images to a fixed input size (e.g., 640x640) and normalize bounding box coordinates to be relative to total width and height of the box. For Faster R-CNN, I'd keep variable image sizes but ensure annotations are in the expected dictionary format. In both cases, I'd expand augmentation (random crops, scale jittering) to improve robustness.

# Reflection / General

What were the key technical challenges you encountered in Day 1 and Day 2?
- figuring out how to get dataset downloaded

What are 2–3 things you’d do differently if you started over?
- better .gitignore file
- more dataset exploration as I went along: look at more images and think about what code is doing

# Overall Steps of Day1  and Day2
I. Project Setup
- Created local and GitHub repo structure (notebooks, data, scripts, requirements.txt)
- Added .gitignore to exclude large dataset files and virtural environment artifacts
- Created and activated a dedicated conda environment (pascalvoc) with dependencies in requirements.txt

II. Dataset Dowload & Inspection (day1_setup.ipynb)
- Downloaded Pascal VOC 2012 dataset
- Verified dataset integrity (number of images and annotation files)
- Inspected variability in images sizes, aspect ratios, and bounding box forms
- Visualized random samples with bounding boxes to ensure correct parsing of annotations

III. Preprocessing & DataLoader (day2_processing.ipynb)
- Built required dataset object using torchvision.datasetsVOCDetection and custom wrapper to modify its output for detection tasks. Custom wrapper included transforms.ToTensor() for image conversion, pleus parxing XML annotations into structioned tensors (boxes, labels) 
- Prepared train/val splits using VOC's offical lists
- Wrote a custom collate_fn for object detection (handles variable images sizes)
- Ran a loop through DataLoader, visualizing sample batches with bounding boxes to verify preprocessing steps

# Day 3 - Image Classification Baseline

Facing challenge during initial training due to a Windows "PicklingError". Using chatGPT to help troubleshoot. On Windows, multiprocessing uses "spawn". Without a guard, PyTorch tries to pickle everything in the global scope, which can fail.