
## High-Level Takeaways

1. **Coverage vs Peak Performance:**  
   - ResNet achieves higher per-class Precision, Recall, and F1 scores for the classes it successfully classifies, but it misses 5 of 20 classes.  
   - YOLO detects all classes, ensuring full coverage, but peak per-class scores are generally lower than ResNet’s.

2. **Class-Specific Strengths:**  
   - ResNet is highly reliable for **people, horses, buses, and dining tables**, with F1 scores approaching 1.0 for some classes.  
   - YOLO performs well on train, motorbike, dining table, cat, and bus in terms of Recall, and on person, cat, and car for Precision.  
   - Critical classes (e.g., people) are better served by ResNet, while YOLO ensures no class is completely missed.

3. **Dataset Limitations:**  
   - Both models struggle on underrepresented classes (bottle, chair, cow, potted plant), highlighting the importance of addressing imbalanced data problems.

4. **Metric Insights:**  
   - F1 is a suitable common metric for comparing ResNet and YOLO due to its strong correlation with YOLO’s mAP (r = 0.92).  
   - The choice of metric should consider project priorities: high Recall is preferable when missing an object is costly, while high Precision is preferable when misclassification is costly.

5. **Practical Considerations:**  
   - ResNet is faster (10–15 min vs 30–50 min for YOLO) and better suited for high-confidence classification of common classes.  
   - YOLO provides full coverage, making it more suitable for applications where detecting all objects is critical, even if some predictions are less accurate.

**Summary:**  
ResNet is optimal for tasks requiring high-confidence detection of common classes, while YOLO is optimal for ensuring all objects are detected. Dataset imbalance limits performance on rare classes, suggesting future work in resampling the training data.





## 1. ResNet Optimization

**Model Parameters:**  
- Batch size: 8  
- Epochs: 10  

**Runtime:** 10–15 minutes  

**Coverage:**  
- Classified 15 out of 20 objects  
- Classes not classified: bottle, cow, potted plant, sheep, sofa  

**Top Classes by Metric:**  
- **ROC Curves:** bus, sheep, cat, horse  
- **Precision:** bus, car, cat, horse, person  
- **Recall:** dining table, person  
- **F1 Score:** person, horse, dining table, cat  

**Observations:**  
- ResNet demonstrates very high per-class scores for the classes it successfully classified, with F1 scores approaching 1.0.  
- Missed classes may be underrepresented in the dataset, contributing to lower coverage.

---

## 2. YOLO Object Detection

**Model Parameters:**  
- Batch size: 8  
- Epochs: 10  

**Runtime:** 30–50 minutes  

**Coverage:**  
- Detected all 20 objects  

**Top Classes by Metric:**  
- **mAP:** train, dog, cat, bus  
- **Precision:** person, cat, car  
- **Recall:** train, motorbike, dining table, cat  
- **F1 Score:** cat, bus, motorbike, person  

**Observations:**  
- YOLO achieves full coverage of all object classes.  
- Per-class scores are generally lower than ResNet’s top scores, with F1 scores mostly below 0.8.  

---

## 3. Model Comparison

**Metric Selection:**  
- F1 is typically used for classification evaluation, while mAP is used for object detection.  
- YOLO’s F1 scores strongly correlate with mAP (r = 0.92, p ≈ 0), making F1 a suitable common metric for comparison with ResNet.  

**Key Insights:**  
1. **Coverage vs Peak Performance:**  
   - ResNet misses 5 classes but achieves higher per-class scores for those it does classify.  
   - YOLO detects all classes but peak scores are lower than ResNet’s.  

2. **Class-Specific Performance:**  
   - Four of ResNet’s missed classes (bottle, chair, cow, potted plant) were among YOLO’s lowest-performing classes, suggesting dataset underrepresentation.  
   - ResNet excels in both **Precision** (horse, bus, car, cat, person) and **Recall** (dining table, person) for several classes.  
   - YOLO excels in Recall for train, motorbike, dining table, and bus, and in Precision for person, cat, and car.  

3. **Scatterplot Analysis:**  
   - ResNet dominates the top Precision and top Recall scores in the recall-vs-precision scatterplot.  
   - Resnet also dominates the bottom Recall scores in the recall-vs-precision scatterplot.  

**Practical Takeaways:**  
- For high-confidence classification (>90% scores), ResNet performs best on:  
  - **Recall:** dining table, person  
  - **Precision:** horse, bus  
- Overall, ResNet is most reliable at classifying **people**, achieving close to 90% for both Precision and Recall and the highest F1 score.  

---

**Referenced Graphics:**  
- `yolo_map_vs_f1_scatter.png` – YOLO mAP vs F1 comparison  
- `resnet_perclass_prf_combined.png` – ResNet per-class Precision/Recall/F1  
- `yolo_perclass_prf1.png` – YOLO per-class Precision/Recall/F1  
- `precision_recall_scatter.png` – Combined ResNet vs YOLO Precision/Recall scatterplot 