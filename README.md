# Sentiment-Analysis-Intepreter

We train a simple transformer for sentiment analysis on movie reviews, extract interpretable features using SAE and generate explanations using LLMs. 

### Downloading the dataset

Visit the link below to download the IMDB Movie Review Dataset for Sentiment Analysis
https://ai.stanford.edu/~amaas/data/sentiment/
Once downloaded, create a folder called 'dataset' in the root directory of this project. From the downoaded dataset, add the 'test' and 'train' folders to the new 'dataset' folder. You can delete any file/folder other than the 'pos' and 'neg' in the 'train' and 'test' folders. The 'dataset' folder should look like this.

dataset/
├── train/
│   ├── pos/
│   └── neg/
└── test/
    ├── pos/
    └── neg/
    
### Setting up the environment

We provide a requirements.txt file to install the packages necessary for this project. Commands below are for setting up a conda environment.

```
conda create --name <env_name>
conda activate <env_name>
pip install -r requirements.txt
```







