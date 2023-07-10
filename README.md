# Surface-Based Retrieval Reduces Perplexity of Retrieval-Augmented Language Models

This repository contains the code for [Surface-Based Retrieval Reduces Perplexity of Retrieval-Augmented Language Models](https://aclanthology.org/2023.acl-short.45/). For setting up the environment, you can use the provided [Singularity](https://docs.sylabs.io/guides/3.7/user-guide/index.html) definition file, or translate it for Docker or Conda. 

## Getting the Prerequisites in Place
We need to have the model, data, and the neighbors ready in place as instructed in [RETRO](https://github.com/TobiasNorlund/retro). You need to run all the steps in that repository before moving forward. 

Then you can run `make retriever_bm25/neighbours.npy` in the `data` directory. The last step can take a while.

## Getting the Results
As soon as you have all the files in place you can run all the steps in `bm25_loss_diff_analysis.ipynb` to get the results. You are going to need a GPU for this step. You might need to change some paths to point to the right files and directories.  

## Citation 
```
@inproceedings{doostmohammadi-etal-2023-surface,
    title = "Surface-Based Retrieval Reduces Perplexity of Retrieval-Augmented Language Models",
    author = "Doostmohammadi, Ehsan  and
      Norlund, Tobias  and
      Kuhlmann, Marco  and
      Johansson, Richard",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-short.45",
    pages = "521--529",
}
```
