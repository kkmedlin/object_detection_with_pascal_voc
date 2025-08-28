##Pascal VOC Project – Development Reflection Questions
# Day 3 - Image Classification Baseline

Facing challenge during initial training due to a Windows "PicklingError". Using chatGPT to help troubleshoot. On Windows, multiprocessing uses "spawn". Without a guard, PyTorch tries to pickle everything in the global scope, which can fail.

1. Dataset & Preprocessing

How do we convert VOC annotations to multi-label targets for the 20 Pascal VOC classes?

Why did we need to handle the single-object vs multiple-object case in encode_voc_target?

Why do we use transforms.Resize and transforms.Normalize before feeding images to ResNet50?

What would happen if the root directory for VOCDetection is set incorrectly?

2. DataLoader

Why did we set num_workers=0 for CPU training?

How does shuffle=True affect training, and why don’t we shuffle validation data?

Why might we subset the dataset during debugging?

3. Model & Transfer Learning

What does param.requires_grad = False do in the context of transfer learning?

Why did we replace model.fc with nn.Linear(..., NUM_CLASSES) and nn.Sigmoid()?

Why do we use Sigmoid instead of Softmax for multi-label classification?

What is the effect of using a smaller model like ResNet18 vs ResNet50 on CPU?

4. Training & Evaluation

Why do we use nn.BCELoss() for multi-label classification?

What is the purpose of optimizer.zero_grad() inside the training loop?

How do we compute ROC AUC for multi-label problems?

Why do we threshold predictions at 0.5 when computing the confusion matrix?

5. Debugging & Performance

Why was training so slow on your CPU before adjustments?

How can you verify that your dataset splits (train / val) are the same across classification and detection tasks?

If we had a CUDA-capable GPU, which sections would benefit the most?