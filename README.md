# cs-175

**Team Name:** Gradient Ascent
**Team Members:** Priya Deshmukh, Davin Makris, Amish Kunal

This repository contains the code, models, and results for our CS 175 final project. The focus of this project is on evaluating different langauge models and techniques for generating Shakespeare-like text using various methods to evaluate which will attain style transfer.

# Repository Structure
```cs-175/
    src/     # contains main code / modules
        classifier/     # judge classfier models
        monkeys/     # generator model architectures (TransformerMonkey, LoRAMonkey, Bigram)
        models/     # saved model checkpoints (.pt files)
        tokenizers/     # tokenizers used by models
        utils/     # data loading 

    experiments/     # experimental scripts
    notebooks/
        final_report.ipynb     # final report notebook
    noise_dataset/     # generated dataset
    results/     # generated figures and qualitative metrics from expirements
    requirements.txt
    README.md
    .gitignore
```
    


# Set up Instructions
1. **Clone the repositiory and switch to <final-submission> branch:**
   ```
    git clone <url>
    cd cs-175
    git checkout final-submission
    ```

3. Install dependencies
   ```
    pip install -r requirements.txt
   ```

5. Run the final notebook:
   
    Open notebooks/final_report.ipynb in Jupyter or VSCode, replace any file paths with your own, and run.
 
    This notebook reproduces figures and results from the final report:
   
    - All models are loaded from checkpoints in src/models/
    - Outputs (figures) are saved in the results/ folder

# Key Modules

- ```src/classifier/```: Contains the TransformerJudge used to evaluate generated text
- ```src/monkeys```: Contains the model architectures used for text generation: TransformerMonkey, LoRAMonkey, BigramModel
- ```src/models/```: Pretrained checkpoints for each model
- ```expirements/```: Scripts used to produce processed results and different approaches in pushing model output to style transfer.
