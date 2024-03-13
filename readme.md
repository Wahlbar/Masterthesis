# Improvement of Open-Set Classification through Adequate Loss Weighting of Negative Samples (Master Thesis)
This github repository contains my implementation and results of the experiments for my master thesis. 
My complete master thesis is available after publication and grading on: https://www.ifi.uzh.ch/en/research/publications/masters-theses.html

As my implementation and experiments is based on the paper of Large-Scale Open-Set Classification Protocols for ImageNet and its github repository, please refer to their README for any further usage and citation: https://github.com/AIML-IfI/openset-imagenet. I do not explain the set-up, dataset and usage of the openset-imagenet repository as this can be read in the above mentioned repository.

## LICENSE
This code package is open-source based on the BSD license.
Please see `LICENSE` for details.

## Setup

We provide a conda installation script to install all the dependencies.
Please run:

    conda env create -f environment.yaml

Afterward, activate the environment via:

    conda activate openset-imagenet

## Scripts

The directory `openset_imagenet/script` includes several scripts, which are automatically installed and runnable.

### Protocols

You can generate the protocol files using the command `imagenet_protocols.py`.
Please refer to its help for details:

    protocols_imagenet.py --help

Basically, you have to provide the original directory for your ImageNet images, and the directory containing the files for the `robustness` library.
The other options should be changed rarely.

### Training of one model

The training can be performed using the `train_imagenet.py` script.
It relies on a configuration file as can be found in `config/train.yaml`.
Please set all parameters as required (the default values are as used in the paper), and run:

    train_imagenet.py [config] [protocol] -o [outdir] -g GPU

where `[config]` is the configuration file, `[protocol]` one of the three protocols, and `[outdir]` the output directory of the trained model and some logs.
The `-g` option can be used to specify that the training should be performed on the GPU (**highly recommended**), and you can also specify a GPU index in case you have several GPUs at your disposal.

### Training of all the models in the paper

The `train_imagenet_all.py` script provides a shortcut to train a model with any of the default loss functions (softmax, softmax with background, entropic open set) as well as any of my additional modification of softmax with background and entropic open set from on three different protocols.
You can run:

    train_imagenet_all.py --loss-functions [list-of-loss-functions] --protocols [list-of-protocols] -g [list-of-gpus]

You can also select some of the `--protocols` to run on, as well as some of the `--loss-functions`, or change the `--output-directory`.
The `-g` option can take several GPU indexes, and trainings will be executed in parallel if more than one GPU index is specified.
In case the training stops early for unknown reasons, you can safely use the `--continue` option to continue training from the last epoch.

Runtime for a single GPU depends heavily on protocols. My experience is the following:


    ```{=latex}

             [PUT LATEX HERE]      

    ```

The following provide a short overview on the different weights used for my loss functions as well as their naming. For a more detailed explanation of my naming please look at my thesis or contact me.


    ```{=latex}

        \begin{table}[t]
            \Caption[tab:overview eos weighting]{Overview of Implemented Entropic Open Set Weighting}{This table provides an overview of the implemented \ac{EOS}  weighting functions of known and negative weights. EOS 1 is the implementation of \cite{palechor2023protocols} with neutral weighting for all samples.}
            \centering
            \begin{tblr}{
              colspec = {cccc},
              cell{1}{2} = {c=3}{c}, % multicolumn
              cell{3}{1} = {r=3}{c}, % multirow
              cell{6}{1} = {r=2}{c}, % multirow
              cell{7}{3} = {c=2}{c}, % multicolumn
              cell{8}{1} = {r=2}{c}, % multirow
              cell{10}{1} = {r=2}{c}, % multirow
              rowsep = 6pt,
              hlines = {black, 0.5pt},
              vlines = {black, 0.5pt}, % vlines can not pass through multicolumn cells
            }
                            & \bf Entropic Open Set (EOS)   &               &                   \\
            Weighting Type  & \bf Name                      & $w_{known}$   & $w_{negative}$    \\ 
            Constant        & \bf EOS 1                     & 1             & 1                 \\
                            & \bf EOS 0.5                   & 1             & 0.5               \\
                            & \bf EOS 0.1                   & 1             & 0.1               \\ 
            Balanced        & \bf EOS BN                    & 1             & \balancednegative \\
                            & \bf EOS BC                    & \balanced     &                   \\      
            Focal           & \bf EOS FM                    & \focalknown   & \focalmax         \\
                            & \bf EOS FS                    & \focalknown   & \focalsum         \\
            Mixed           & \bf EOS FK                    & \focalknown   & \balancednegative \\
                            & \bf EOS FN                    & \balanced     & \focalmax         \\
            \end{tblr}
        \end{table}    

        \begin{table}[t]
            \Caption[tab:overview bg weighting]{Overview of Implemented SoftMax Weighting}{This table provides an overview of the implemented SoftMax with Background (BG)  weighting functions of known and negative weights. The 'balanced' BG B was already implemented by \cite{palechor2023protocols}. I reused it as a baseline.}
            \centering
            \begin{tblr}{
              colspec = {cccc},
              cell{1}{2} = {c=3}{c}, % multicolumn
              cell{3}{3} = {c=2}{c}, % multicolumn
              cell{4}{3} = {c=2}{c}, % multicolumn
              cell{5}{3} = {c=2}{c}, % multicolumn
              cell{6}{1} = {r=2}{c}, % multirow
              rowsep = 6pt,
              hlines = {black, 0.5pt},
              vlines = {black, 0.5pt}, % vlines can not pass through multicolumn cells
            }
                            & \bf SoftMax with Background (BG)  &               &                   \\
            Weighting Type  & \bf Name                          & $w_{known}$   & $w_{negative}$    \\ 
            Constant        & \bf BG 1                          & 1             &                   \\ 
            Balanced        & \bf BG B                          & \balanced     &                   \\    
            Focal           & \bf BG F                          & \focalknown   &                   \\
            Mixed           & \bf BG FK                         & \focalknown   & \balancednegative \\
                            & \bf BG FN                         & \balanced     & \focalknown       \\

            \end{tblr}
        \end{table} 

    ```

### Evaluation

After the training the protocols use `evaluate_imagenet.py` to evaluate training protocols.

You can run:

    evaluate_imagenet.py --loss [loss-function] --protocol [protocols] -g

where `-b` can be used to evaluate the best model. Without it the last model is used for the evaluation.

Finally, the `plot_imagenet.py` script can be used to perform the plots.

You can run:

    plot_imagenet.py --loss-functions [list-of-loss-functions] --labels [list-of-labels] --protocols [list-of-protocols] -g [list-of-gpus]

Similar as for the evaluation --use-best can be used to evaluate the best model.
For further explanation and functionalities refer to the openset imagenet github: https://github.com/AIML-IfI/openset-imagenet.


This script will use all trained models (as resulting from the `train_imagenet_all.py` script), extract the features and scores for the validation and test set, and plots into a single file (`Results_last.pdf` by default), as well as providing the table from the appendix as a LaTeX table (default: `Results_last.tex`)

1. OSCR curves are generated on the negative and unknown samples.
2. Confidence propagation plots on negative and known samples during training are calculated and plotted.
3. Histograms of softmax scores of known, negative and unknown samples are generated.

Please specify the `--imagenet-directory` so that the original image files can be found, and select an appropriate GPU index.
You can also modify other parameters, see:

    plot_imagenet.py --help

## Getting help

In case of trouble, feel free to contact me under [simon.giesch@gmail.com](mailto:simon.giesch@gmail.com)
