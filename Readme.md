This repo is under development.

This is a repo for pretraining T5Mem model on Masked language modeling
## Install requirements
This repo is based on 🤗 Transfomers implementation of the T5 model.
Horovod is used for multi-gpu and multi-node training.


###  Install Horovod
For the same steup as mine you can follow these steps:
1. Install [conda](https://docs.anaconda.com/anaconda/install/index.html)
2. run this commands inside the project folader  
   ```# set relevant build variables for horovod
        export ENV_PREFIX=$PWD/env
        export CUDA_HOME=$ENV_PREFIX
        export NCCL_HOME=$ENV_PREFIX
        export HOROVOD_CUDA_HOME=$CUDA_HOME
        export HOROVOD_NCCL_HOME=$NCCL_HOME
        export HOROVOD_NCCL_LINK=SHARED
        export HOROVOD_GPU_OPERATIONS=NCCL```        
      then run the command:        
        ```
        conda env create --prefix $ENV_PREFIX --file horovod.yml --force
        ```
3. Activate the conda environment after that run one by one pip installations:
   ```
   pip  install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
   pip  install huggingface-hub==0.0.12
   pip  install transformers==4.8.2
   pip  install einops==0.3.2
   pip  install prettytable==2.1.0
   pip  install sacrebleu
   pip  install sentencepiece
   pip  install datasets==1.8.0
   pip  install horovod[pytorch]==0.22.1
  
   ```
4. Run the last command 
        ```
        horovodrun --check-build
        ```
You should see output similar to the following.:
```
Horovod v0.22.1:                                                                                                                                                                                           │·······································································
                                                                                                                                                                                                           │·······································································
Available Frameworks:                                                                                                                                                                                      │·······································································
    [ ] TensorFlow                                                                                                                                                                                         │·······································································
    [X] PyTorch                                                                                                                                                                                            │·······································································
    [ ] MXNet                                                                                                                                                                                              │·······································································
                                                                                                                                                                                                           │·······································································
Available Controllers:                                                                                                                                                                                     │·······································································
    [X] MPI                                                                                                                                                                                                │·······································································
    [X] Gloo                                                                                                                                                                                               │·······································································
                                                                                                                                                                                                           │·······································································
Available Tensor Operations:                                                                                                                                                                               │·······································································
    [X] NCCL                                                                                                                                                                                               │·······································································
    [ ] DDL                                                                                                                                                                                                │·······································································
    [ ] CCL                                                                                                                                                                                                │·······································································
    [X] MPI                                                                                                                                                                                                │·······································································
    [X] Gloo
```
You may need to install some libraries install them using pip
       
   ✨ Vioal ✨ ! you are now ready to reproduce the experiements!
run the experiement:
```
sh run_mlm.sh
```
 to draw attention maps rerun the [notebook](https://github.com/Arij-Aladel/T5-Tasks/blob/main/T5-heatmap_MLM_test_128.ipynb)
