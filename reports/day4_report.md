##Pascal VOC Project – Development Reflection Questions
# Day 3 - Object Detection

# Understanding the Workflow

How does converting Pascal VOC XML annotations to YOLO format affect the training process? What are the key differences between these formats?

When you subset the dataset for CPU testing, what trade-offs are you making in terms of model performance and debugging speed?

How does image size (imgsz) influence both detection quality and inference speed? Did you notice any patterns today?

# Model Performance & Predictions

Why were your initial predictions all “no detections”? How did adjusting the model or dataset change that?

What factors could lead YOLO to detect some objects and miss others, even when the bounding boxes exist in the dataset?

When running predictions for portfolio images, what did you learn from comparing the original images with YOLO predictions?

# Code & Data Management

How did your .gitignore changes help manage large files and prevent unnecessary data from being tracked?

In your current code, how is the predicted image path determined? Could there be a simpler or more robust approach?

What errors did you encounter when switching from the subset to the full dataset, and what does that teach you about using Python modules correctly (e.g., glob)?

# Next Steps & Improvements

If you wanted to create a larger, high-quality portfolio of predictions, what changes would you make to the current workflow?

How could you automate the process of generating and saving side-by-side images for all validation images without running into the “prediction not found” issue?

What would be your strategy for training on the full dataset without overwhelming your CPU resources?