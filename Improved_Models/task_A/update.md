# Analysis of `task-a-divy-v2.ipynb` Results

## Findings / Inferences

1. **HPO Sweep Collapse (Phase 1):** 
   - Across all 8 Optuna trials, the validation Macro F1 score flatlined at **0.3227**. 
   - This exact score corresponds to the model acting essentially as a majority/constant class classifier. Learning completely collapsed down to random guessing or static output in every parameter config.

2. **Full Training Phase (Phase 2):**
   - The validation result matched the collapsed HPO result. It predicted the `human` class with 100% recall and 0% precision for `machine`—meaning the model effectively predicted class `human` for *every entirely single example*.
   - Overall Validation Accuracy was 0.48, Macro F1: **0.3227**.

3. **Holdout Test Set Evaluation:**
   - On the final 1k test sample (`task_a_test_set_sample.parquet`), it exhibited the exact same collapsed behavior. Accuracy: 0.7770, Macro-F1: **0.4373**. Recall for `1` (Machine) was 0.00.
   - It only achieved high accuracy because the sample set happens to be heavily imbalanced toward class 0 (Human - 777 samples out of 1000).

## Root Cause
The GraphCodeBERT model experienced **Gradient Vanishing/Explosion or Catastrophic Forgetting** almost immediately. Because all trials got stuck at the ~0.32 Macro F1 mark predicting a single class, the learning rate boundaries in the config space might be too high or the initial weights of the newly added layers (or frozen layers) are sabotaging the gradients. It's essentially a local minimum of just guessing "Human".

## Recommended Improvements for Next Run
1. **Reduce Learning Rate:** The current range bounds may still be too high for this backbone. Try setting an even smaller range or a very slow warmup scheduling strategy.
2. **Increase Class Weights logic:** Ensure the CrossEntropyLoss is using class weights correctly. If it just collapses to the majority class over predicting, balancing the dataset or increasing minority class logit weights is required.
3. **Unfreeze bottom layers gently (Gradual Unfreezing):** Sometimes freezing 4-8 layers of a transformer while attaching an untrained linear head scrambles the representations too quickly. It needs a high learning rate on the head, but a tiny one on the transformer.
4. **Gradient Clipping:** Ensure `max_grad_norm` is actively clipping during the optimization loop to prevent explosion.
