# Project introduction

-   Main goal - to make a model that will be able to classify text (unstructured data) reviews as negative (aka 0) or positive (aka 1)

# How to quickly review the result of project

1. go to `final_test.ipynb` file and see the results (scroll to the end of file)
2. then you can go to `notebooks/paral_conv_linear.ipynb` to look more deeply on how I trained the-final-choice model most reliable model with approximately 93% accuracy on filtered and sampled data
3. Then you can go even deeper - go to `src/architectures` and explore
   `paral_conv_linear_arch1.py` and its submodule - `parallel_1dconvs_layer.py` in `src/architectures/submodules/`

# Results

93% accuracy on filtered and sampled data

# Final model description

### Final model

Parallel1DConvsLinearClass. Parallel convolutions layer => flatten layer => multiple fully connected (linear) layers => sigmoid activation function for output.

# Used tools

-   for data processing - custom functions written in python (src/data_preprocessing/)
-   for transforming texts into matrices - word embeddings by spacy's 'en_core_web_lg'
-   utils/live_plot.py - is just minitool written by myself for easy visualisation of cost(loss) function value changes in real time (during training) so I could have more in-depth understanding of what's happening during training process (it really helps).
