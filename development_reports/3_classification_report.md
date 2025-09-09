##Pascal VOC Project – Development Reflection Questions
# Day 3 - Image Classification Baseline

Facing challenge during initial training due to a Windows "PicklingError". Using chatGPT to help troubleshoot. On Windows, multiprocessing uses "spawn". Without a guard, PyTorch tries to pickle everything in the global scope, which can fail.

1. Dataset & Preprocessing

How do we convert VOC annotations to multi-label targets for the 20 Pascal VOC classes?
converted annotions into multi-hot vectors. multi-hot vectors are vectors made up of 0s and 1s indicated which annotations are contained in each single image.

Why did we need to handle the single-object vs multiple-object case in encode_voc_target?
multi-hot vectors are used for identifying images with multiple objects. in contrast, single-hot vectors are used for identifying images with or without a single object (only for binary datasets)

Why do we use transforms.Resize and transforms.Normalize before feeding images to ResNet50?
batches of tensors need to be the same size - reason for resizing. Normalization puts pixel values on the same comparative RGB scale. 

What would happen if the root directory for VOCDetection is set incorrectly?
if the root directory for VOC Dectection is set incorrectly, the data won't get loaded in and VOCDetection will throw an error

2. DataLoader

Why did we set num_workers=0 for CPU training?
Windoes isn't set up for more than 0 workers. Causes pickling issues. Would have worked fine on Linux or mac machine.  

How does shuffle=True affect training, and why don’t we shuffle validation data?

shuffling training data helps with preventing overfitting be introducing randomization in sampling the training data. We don't shuffle validation data as we use all validation data (that's kept entirely separate from training process) to ensure more robust and accurate evaluation results

Why might we subset the dataset during debugging?
To use less data. Speeds up the process while we're trying to get the code to run.

3. Model & Transfer Learning

What does param.requires_grad = False do in the context of transfer learning?
param.requires_grad = False freezes pretrained layers so they're not updated during backpropagation, saving compute and preserving pretrained features. 

Why did we replace model.fc with nn.Linear(..., NUM_CLASSES) and nn.Sigmoid()?
We replaced model.fc with nn.Linear(..., NUM_CLASSES) and nn.Sigmoid() so that the model matched the dataset's number of classes (20). ImageNet has 1000 so we had to change the final connected layer.

Why do we use Sigmoid instead of Softmax for multi-label classification?
Softmax is for determining classification of binary data (sums up to 1). Sigmoid is used instead of Softmax as it allows for generating results for multi-class data: indentifying which of the multi-classes are likely in the image (each class gets a 0 to 1 probability).

What is the effect of using a smaller model like ResNet18 vs ResNet50 on CPU?
Less layers and parameters so it's less computationally expensive. So a CPU is able to handle it. ResNet18 may generate performance results weaker than those of ResNet50.

4. Training & Evaluation

Why do we use nn.BCELoss() for multi-label classification?
BCE stands for Binary Cross Entropy Loss. This loss function allows for computing loss with respect to multiple classes. It relies on multi-variate calculus and takes the derivative w.r.t. each class. Then it averages these losses for an overall loss.

What is the purpose of optimizer.zero_grad() inside the training loop?
It's important to zero-out your gradient before each learning cycle. If you don't zero them, updates would be incorrect.

How do we compute ROC AUC for multi-label problems?
Computes truly positive rate and falsely positive rate for each class and then averages them. it does this over different tresholds (0 to 1). When looking at a plot of these, the top-left is the best result.

Why do we threshold predictions at 0.5 when computing the confusion matrix?
0.5 is a base score: you have a 50/50 chance of guessing correctly when youre choice is 0 or 1.

5. Debugging & Performance

Why was training so slow on your CPU before adjustments?
Because the code was set up to run on GPU -- the number of workers, the model -- these things were more complex and computationally expensive that what my CPU could handle.

How can you verify that your dataset splits (train / val) are the same across classification and detection tasks?
You can print out a test to confirm they're the same.

If we had a CUDA-capable GPU, which sections would benefit the most?
The model section as model training involves matrix multiplicaiton as data is passed forwards and backwards over the model.