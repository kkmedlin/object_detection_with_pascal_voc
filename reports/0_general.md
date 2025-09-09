##Pascal VOC Project – Development Reflection Questions
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