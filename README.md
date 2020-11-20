# interpret-math-transformer
Data, model weights, and code for analyzing and visualizing representations learned by transformer models trained on Mathematics Dataset. 

Analyses were conducted on models that were trained on the [Mathematics Dataset](https://arxiv.org/pdf/1904.01557.pdf), which can be found [here](https://github.com/deepmind/mathematics_dataset). 

The entire dataset is not necessary for reproducing our experiments, as we were analyzing models trained in [previous work](https://arxiv.org/abs/1910.06611), the code for which can be found [here](https://github.com/ischlag/TP-Transformer). The customized test sets we created can be found in the data/ directory. 

The two pre-trained models used in our experiments can be found [here](https://sandbox.zenodo.org/record/699942#.X7f-6ZNKjLB).


After downloading the weights, to run the analyses on query vectors only, for TP-Transformer:
```
analyze.py --return_Q --model_name TP --load_model weights/TP_Transformer.pt
```

Standard transformer:
```
analyze.py --return_Q --model_name TF --load_model weights/Transformer.pt
```

The results can be visualized with the jupyter notebook found in the notebooks directory. 

Linear regression analyses can be performed with the regression.Rmd file in the results/ directory. 