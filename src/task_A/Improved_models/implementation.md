# GraphCodeBERT Implementation Plan & Analysis

## 1. Insights

1. **Actually Use GraphCodeBERT's Graph Features (Critical)**
   - Parse code into AST (tree-sitter), extract DFG (Data Flow Graph) nodes.
   - Inject DFG into inputs with custom `attention_mask` and `position_ids`.
   
2. **Increase Sequence Length via Gradient Checkpointing**
   - Increase `MAX_LENGTH` to 512 and enable `gradient_checkpointing=True` to offset memory usage on the Kaggle T4.

3. **Implement Leave-One-Language-Out (LOLO) Validation**
   - Train on two languages (e.g., Python, Java) and validate on the third (e.g., C++) to gauge generalization to unseen languages.

4. **Advanced Loss Functions (Focal Loss)**
   - Use Focal Loss to heavily penalize hard, borderline cases between advanced AI generators and human code.

5. **Multi-Sample Dropout**
   - Wrap the Sequence Classification head in multiple dropout layers and average the logits to stabilize training and improve robustness.

---

## 2. Pondering & Feasibility for Kaggle T4

- **Data Flow Graph (DFG) Extraction**: **(Status: VERY HARD for on-the-fly Notebook)**
  - *Why?* Requires `tree-sitter`, compiling language parsers (`.so` files for Java, Python, C++), and running AST traversals for 500K snippets over CPU. In a Kaggle environment, this takes many hours and is highly prone to kernel timeouts.
  - *Mitigation*: It is highly recommended to do DFG extraction offline as a separate data preparation step, upload it as a Kaggle dataset, and then use it. In the main notebook, doing full dynamic tokenization with DFG is too heavy. I will skip full AST extraction in this immediate notebook kernel but leave placeholders, focusing instead on model robustness points.
  
- **Gradient Checkpointing & MAX_LENGTH=512**: **(Status: HIGHLY FEASIBLE)**
  - *Why?* HuggingFace `TrainingArguments` supports this out of the box. Extremely useful for dealing with long algorithmic competitive programming solutions.
  
- **Leave-One-Language-Out (LOLO) Validation**: **(Status: HIGHLY FEASIBLE)**
  - *Why?* We already have the metadata. We can split logic based on the `language` column instead of random stratified splits.
  
- **Focal Loss**: **(Status: HIGHLY FEASIBLE)**
  - *Why?* Subclassing `Trainer` and overriding `compute_loss` is straightforward and adds no computational overhead.
  
- **Multi-Sample Dropout Head**: **(Status: HIGHLY FEASIBLE)**
  - *Why?* Creating a custom PyTorch `nn.Module` using `RobertaModel` is simple and adds minimal overhead.

## 3. Implementation Plan
I will implement Points 2, 3, 4, and 5 directly in the `graphcodebert_task_a.ipynb` notebook. Point 1 will be deferred to an offline dataset building process to avoid Kaggle CPU timeouts.