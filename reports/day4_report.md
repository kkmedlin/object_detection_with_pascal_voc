##Pascal VOC Project – Development Reflection Questions
# Day 4 - Object Detection

# Understanding the Workflow

1. How does converting Pascal VOC XML annotations to YOLO format affect the training process? What are the key differences between these formats?

Pascal VOC XML annotations are absolute pixel values. For YOLO, you want relative coordinates instead. This means you need to normalize images during preprocessing before YOLO. This makes it possible for YOLO to handle images of differing sizes.

2. When you subset the dataset for CPU testing, what trade-offs are you making in terms of model performance and debugging speed?

I am trading speed for accuracy. Training with the full dataset takes way too long for my goal of learning how the tools work. With a smaller subset, however, the model is at greater risk of overfitting onto the training set and, therefore, be less able to correctly detect images in the validation set. In other words, the training set is less likely to be repesentative to the overall dataset.

3. How does image size (imgsz) influence both detection quality and inference speed? Did you notice any patterns today?

Greater the image size, the greater the resolution. Therefore, it's easier to see smaller objects and easier to discern what they are. However, greater resolution slows down inference(= detection + classification). The more pixels there to process, the longer it takes to compute an image's forward pass over a model -- what's involved in inference.

# Model Performance & Predictions

4. Why were your initial predictions all “no detections”? How did adjusting the model or dataset change that?

Initially, probability thresholds for predictions were set higher (0.5) and training dataset size was quite small. By decreasing the threshold to 0.1, we allowed YOLO to make a prediction even if it was likely incorrect. Additionally, by increasing the training set size, we likely decreased overfitting and therefore improved our chances of YOLO resulting in correct object detection.

To note: By decreasing thresholds and increasing training data, I made YOLO less sensitive to poor anchor priors. In YOLO, anchors are prior bounding box shapes that the model uses as an initial guess for object detection. YOLO has a built in process of automatically detecting poor bounding box priors. 

5. What factors could lead YOLO to detect some objects and miss others, even when the bounding boxes exist in the dataset?

The objects represented in the dataset more often will be easier to detect that those under-repesented in the dataset -- simply because the model is undertrained on those minority represented objects. Bounding boxes show where an object is in an image; not what it is.

Additionally, when objects are close together within an image, mistakes can happen. To deter from mistakes, YOLO uses Bounding Box Regression to predict adjustments to anchor boxes -- making them more closely match true bounding boxes in training set. Additioanlly, YOLO handles overlapping bounding boxes per object with Non-Max Suppression (NMS). It sorts them by confidence scores, keeps the best one, and removes others with IoU > threshold. While this addresses multiple labels per object; it doesn't address when proximate objects have overlapping bounding boxes. This can lead to detecting some objects and missing others.

6. When running predictions for portfolio images, what did you learn from comparing the original images with YOLO predictions?

YOLO predictions are better when the objects in the image are less crowded together. It's easier to decipher images that are disjoint from one another.

7. What metric is most commonly used to evaluate YOLO’s predictions? How does mAP compare to F1 score in classification? 

Traditionally, F1 is uesed to compare different classification models and mAP is used to compare different detection (localization + classification) models. So if those metrics are used, it's a bit like comparing apples and oranges, though they are both on the 0 to 1 scale.

However, you could do a precision/recall score per class to do a raw side-by-side comparison.

# Code & Data Management

8. How did your .gitignore changes help manage large files and prevent unnecessary data from being tracked?

It was awesome. As soon as I realized VSCode was tracking 10s of thousands of new files from the data preprocessing step for YOLO, I added more files for gitbut to ignore so that it wouldn't get bogged down.

9. In your current code, how is the predicted image path determined? Could there be a simpler or more robust approach?

It's predicting the first five images in the validation set. A more robust approach could involve doing a random sampling for qualitative checks and generating maP scores from the full validation set.

10. What errors did you encounter when switching from the subset to the full dataset, and what does that teach you about using Python modules correctly (e.g., glob)?

The glob module provides a way to find path names matching a specific patern, which was super helpful for searching for and finding XML and JPG files in the original Pascal VOC dataset in order to preprocess it for YOLO. (made image sizes the same, normalized bounding boxes)

When I switched between types of datasets, I had mistakenly used glob(path) instead of glob.glob(path) -- I needed the glob.glob to call the glob function instead of the glob module.

# Next Steps & Improvements

11. If you wanted to create a larger, high-quality portfolio of predictions, what changes would you make to the current workflow?

I would run it on a GPU instead of a CPU so have a more robust model. I would use more images for training the model. I would increase the threasholds for predictions.

12. How could you automate the process of generating and saving side-by-side images for all validation images without running into the “prediction not found” issue?

I could write a script that loops over validation images, runs predictions, and saves both original and predicted images in a paired directory. So that it keeps going even when there's a "prediction not found" prediction. Then you could have it generate side-by-side images, skipping over the ones without predictions.

13. What would be your strategy for training on the full dataset without overwhelming your CPU resources?

The YOLO model I'm using has been pretrained on COCO -- a very large dataset with 80 images, including those in Pascal VOC. Therefore, the early layers of YOLO already extract useful features (edges, textures, shapes). So we can decrease overfitting without increasing computational load by freezing layers (prevent the early layers from updating) and fine-tuning (only train the later layers that specialize in Pascal VOC's objects). With these strategies--freezing layers and fine-tuning-- we can hopefully improve results without using larger training sets. 