from .model import GPT, GPTConfig
from .dataset import NameDataset, CharCorruptionDataset
from .trainer import Trainer, TrainerConfig

import torch
import random
random.seed(0)

def initialize_vanilla_model(mconf):
    attention_model = None
    ### TODO:
    ### [part c]: Make some model here

    ### START CODE HERE
    attention_model = GPT(mconf)
    ### END CODE HERE
    return attention_model

def initialize_perceiver_model(mconf, bottleneck_dim=32):
    attention_model = None
    ### TODO
    ### [part g]: Make some other model here

    ### START CODE HERE
    perceiver_config = GPTConfig(
        vocab_size=mconf.vocab_size,
        block_size=mconf.block_size,
        perceiver=True,
        bottleneck_dim=bottleneck_dim,
        n_layer=mconf.n_layer,
        n_head=mconf.n_head,
        n_embd=mconf.n_embd,
        embd_pdrop=mconf.embd_pdrop,
        resid_pdrop=mconf.resid_pdrop,
        attn_pdrop=mconf.attn_pdrop
    )
    attention_model = GPT(perceiver_config)
    ### END CODE HERE
    return attention_model

def finetune(reading_params_path, finetune_corpus_path, pretrain_dataset, block_size, model, finetune_lr=6e-4, writer=None):
    ### TODO:
    ### [part c] [part f]:
    ### - Given:
    ###     1. A finetuning corpus specified in finetune_corpus_path
    ###     2. A path reading_params_path containing pretrained model
    ###         parameters, or None if finetuning without a pretrained model
    ### - Goals:
    ###     1. If reading_params_path is specified, load these parameters
    ###         into the model
    ###     2. Finetune the model on this corpus
    ###
    ### - Make sure to use the following hyperparameters:
    ###     Hyperparameters for finetuning WITHOUT a pretrained model:
    ###         max_epochs=75
    ###         batch_size=256
    ###         learning_rate=6e-4
    ###         lr_decay=True
    ###         warmup_tokens=512*20
    ###         final_tokens=200*len(pretrain_dataset)*block_size
    ###         num_workers=4
    ###     Hyperparameters for finetuning WITH a pretrained model:
    ###         max_epochs=10
    ###         batch_size=256
    ###         learning_rate=6e-4
    ###         lr_decay=True
    ###         warmup_tokens=512*20
    ###         final_tokens=200*len(pretrain_dataset)*block_size
    ###         num_workers=4
    ###
    ###
    ### Note: Please use torch.load(reading_params_path, map_location=torch.device('cpu')) to load pretrained model 

    trainer_obj = None #Trainer object (see trainer.py for more details)
    tconf = None #TrainerConfig object (see trainer.py for more details)
    ### START CODE HERE
    # Load pretrained parameters if reading_params_path is specified
    if reading_params_path:
        pretrained_params = torch.load(reading_params_path, map_location=torch.device('cpu'))
        model.load_state_dict(pretrained_params)
        
        # Set hyperparameters for finetuning with a pretrained model
        max_epochs = 10
    else:
        # Set hyperparameters for finetuning without a pretrained model
        max_epochs = 75
    
    # Create dataset for finetuning
    train_dataset = NameDataset(pretraining_dataset=pretrain_dataset, data=open(finetune_corpus_path, encoding='utf-8').read())
    
    # Initialize TrainerConfig
    tconf = TrainerConfig(max_epochs=max_epochs, batch_size=256, learning_rate=finetune_lr, lr_decay=True,
                          warmup_tokens=512*20, final_tokens=200*len(pretrain_dataset)*block_size,
                          num_workers=4)
    
    # Initialize Trainer object
    trainer_obj = Trainer(model, train_dataset=train_dataset, test_dataset=None, config=tconf)
    #trainer_obj.train()
    ### END CODE HERE
    return tconf, trainer_obj

def pretrain(pretrain_dataset, block_size, model, pretrain_lr=6e-3, writer=None):
    ### TODO:
    ### [part f]:
    ### - Given:
    ###     1. A corpus specified in pretrain_dataset
    ### - Goals:
    ###     1. Pretrain the model on this corpus
    ###
    ### - Make sure to use the following hyperparameters for pretraining:
    ###     max_epochs=650
    ###     batch_size=128
    ###     learning_rate=6e-3
    ###     lr_decay=True
    ###     warmup_tokens=512*20
    ###     final_tokens=200*len(pretrain_dataset)*block_size
    ###     num_workers=4

    trainer_obj = None #Trainer object (see trainer.py for more details)
    tconf = None #TrainerConfig object (see trainer.py for more details)

    ### START CODE HERE
    #train_dataset = CharCorruptionDataset(data=pretrain_dataset, block_size=block_size)
    tconf = TrainerConfig(max_epochs=650, batch_size=128, learning_rate=pretrain_lr, lr_decay=True, warmup_tokens=512*20,
                          final_tokens=200*len(pretrain_dataset)*block_size, num_workers=4)
    
    trainer_obj = Trainer(model, train_dataset=pretrain_dataset, test_dataset=None, config=tconf)
    #trainer_obj.train()
    ### END CODE HERE
    return tconf, trainer_obj

def train(model, writing_params_path, trainer_obj):
    ### TODO:
    ### - Given:
    ###     An output path writing_params_path for the model parameters
    ### [part c]:
    ###
    ### Note: trainer_obj is of type Trainer (see trainer.py for more details)

    ### START CODE HERE
    model_params = torch.load(writing_params_path, map_location=torch.device('cpu'))
    model.load_state_dict(model_params)
    trainer_obj.train()
    ### END CODE HERE
    return
