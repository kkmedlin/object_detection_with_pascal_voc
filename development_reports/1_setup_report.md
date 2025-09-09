##Pascal VOC Project – Development Reflection Questions
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