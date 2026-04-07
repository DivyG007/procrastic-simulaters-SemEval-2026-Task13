# Analysis of `task-a-divy-v2.ipynb` Results

## 1. Total Model Collapse (Majority Class Prediction)
* **HPO Phase:** Almost every single Optuna trial resulted in a Validation Macro F1 score of exactly **0.3227**. 
* **Validation Phase (Full Training):** The final model scored an Accuracy of 0.48 and a Macro F1 of **0.3227**. The classification report shows it predicted the `human` class with a recall of 1.00 and precision of 0.48, while the `machine` class had a recall of 0.00 and precision of 0.00. 
* **Test Set Phase:** On the final out-of-distribution test set, the accuracy was 0.78, but the Macro F1 was only **0.43**. Again, the precision and recall for the `machine` class (class 1) was **0.00**.

**Inference:** The model learned absolutely nothing. It experienced what is called "mode collapse" or "majority class collapse". It essentially figured out that the easiest way to minimize loss (or due to gradient explosion) was to just predict "human" for every single code snippet.

## 2. Root Causes of the Collapse
There are a few key reasons why GraphCodeBERT failed to learn and collapsed immediately:
1. **Severe Class Imbalance Ignored:** If your dataset is heavily skewed (which the test set implies, 777 human vs 223 machine), the `CrossEntropyLoss` will naturally bias towards the majority class. Although you used a `FocalLossTrainer` to help with this, it either wasn't enough or the `gamma`/`alpha` parameters weren't tuned correctly for this specific imbalance.
2. **Learning Rate Too High for Transformer backbone:** The Optuna search bounded the learning rate between 5e-6 and 3e-5. If you pair this with freezing bottom layers and a relatively small batch size without proper gradient clipping, transformers often experience catastrophic gradient explosions right in the first epoch, permanently breaking the weights.
3. **Weight Decay & Dropout combinations:** Standard regularization techniques like high dropout (0.25+) applied locally to a totally uninitialized head on top of a frozen set of transformer layers often prevent the head from successfully updating.

## Recommendations for the next run:
1. **Compute and Apply Class Weights**: Feed explicit `class_weights` inversely proportional to class frequencies into the `CrossEntropyLoss` to force the model to penalize machine-class mistakes heavily.
2. **Implement Gradient Clipping**: Add `max_grad_norm=1.0` in your `TrainingArguments` to prevent the gradients from exploding during the warmup phase.
3. **Adjust the Focal Loss parameters**: If keeping Focal Loss, you may want to increase `gamma` (e.g., to 3.0) to force the model to focus on hard-to-predict examples (the machine class), and tune `alpha` to explicitly weight the minority class higher.
4. **Lower the Learning Rate Space**: Drop the Optuna LR upper bound slightly (e.g., `1e-6` to `2e-5`).
