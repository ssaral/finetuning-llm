# Finetuning-llm
Generic fine-tuning LLM model


## Explanation:
- The python files will give you full control on hyperparameters, model, dataset to be used to fine-tuning the model (can be from hugging-face or either in JSON/CSV format). For specifc data-format, you can make some minor edit the script to accomodate you training setup.
- The content is split into clear sections: **Installation Steps** and **Example Command**, each with its own explanation and code snippet.


## Installation and Example Usage of `lm-eval`

### Installation Steps

To install the `lm-eval` package, follow these steps:

1. Clone the repository:
    ```bash
    git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
    ```

2. Navigate to the project directory:
    ```bash
    cd lm-evaluation-harness
    ```

3. Install the package:
    ```bash
    pip install -e .
    ```

### Example Command

After installing, you can run the following example command to evaluate a model:

```bash
lm_eval --model hf \
    --model_args pretrained=EleutherAI/pythia-160m,revision=step100000,dtype="float" \
    --tasks lambada_openai,hellaswag \
    --device cuda:0 \
    --batch_size 8
