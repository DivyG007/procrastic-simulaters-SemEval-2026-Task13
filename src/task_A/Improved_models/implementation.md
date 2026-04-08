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
## 4. Completed Implementations & Fixes (Session Log)

During the current debugging and optimization session, several API incompatibilities, performance bottlenecks, and CUDA multiprocessing issues were resolved in `graphcodebert_task_a.ipynb`.

### 1. Multi-Sample Dropout Model & Trainer API Fixes
- **`supports_gradient_checkpointing`**: Explicitly added `supports_gradient_checkpointing = True` to the custom `GraphCodeBERTMultiDropModel` wrapper to allow Hugging Face's `Trainer` to utilize VRAM-saving techniques.
- **Output API Conformity**: Modified the custom `forward` method to explicitly return a `SequenceClassifierOutput` object, ensuring compatibility with the standard `Trainer` expectations instead of returning raw tuples/dictionaries.
- **`compute_loss` Resiliency**: Updated the overridden `FocalLossTrainer.compute_loss()` to absorb `**kwargs` (specifically `num_items_in_batch` introduced in newer Hugging Face versions), preventing pipeline crashes during the inner training loop.
- **Trainer Initialization**: Removed the `tokenizer` kwargs from the `FocalLossTrainer` instantiation, as `data_collator` already manages it, resolving a `TypeError` signature mismatch.

### 2. Multiprocessing & CUDA Crash Resolution
- **CUDA Fork Issue**: Removed `num_proc=4` from the `Dataset.map` calls. Spawning multiple processes *after* the model was initialized and moved to the GPU (`model.to('cuda')`) caused an unrecoverable `RuntimeError: Cannot re-initialize CUDA in forked subprocess`. 

### 3. Ultimate Training Speed Optimizations
- **Micro-Batching Over Checkpointing**: Set `gradient_checkpointing=False` entirely to avoid the +30% computation time overhead of re-calculating the forward pass during backpropagation.
- **Maintained Effective Batch Size**: Dynamically partitioned the batches: halved `per_device_train_batch_size` (e.g., from 16 to 8) and doubled `gradient_accumulation_steps` (e.g., from 2 to 4). This stays within the Tesla T4's 16 GB VRAM limit while maintaining the exact same effective batch size (32).
- **Evaluator Hook Efficiency**: Switched `eval_strategy` and `save_strategy` to `"epoch"` to eliminate the heavy latency penalty caused by stopping multiple times mid-epoch to evaluate.
- **Data Throughput Optimizations**: Added `dataloader_num_workers=4` and `dataloader_pin_memory=True` to maximize CPU-to-GPU data transfer speeds.
- **Half-Precision Eval**: Enabled `fp16_full_eval=True` to optimize the evaluation loop memory and compute footprint.
- **Fused Optimizer**: Upgraded to `optim="adamw_torch_fused"` for a faster, optimized C++ backend ADAMW implementation.
