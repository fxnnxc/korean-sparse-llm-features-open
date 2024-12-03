import pickle
import torch 
import os 
import seaborn as sns 
from tqdm import tqdm 
import matplotlib.pyplot as plt 
import numpy as np
from nlp_features.sae.standard_sae import StandardTrainer, AutoEncoder
from nlp_features.sae.gated_sae import GatedTrainer, GatedAutoEncoder
from nlp_features.sae.gated_anneal_sae import GatedAnnealTrainer
from nlp_features.sae.standard_anneal_sae import StandardAnnealTrainer

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, activations):
        activations = activations.view(-1, activations.shape[-1])
        self.activations = activations.float()
    def __len__(self):
        return len(self.activations)
    def __getitem__(self, idx):
        return self.activations[idx]

print()

import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('--lm_name', type=str, default='exaone')
parser.add_argument('--lm_size', type=str, default='8b')
parser.add_argument('--split', type=str, default='train')
parser.add_argument('--sae', type=str, default='standard')
parser.add_argument('--lang', type=str, default='en')
parser.add_argument('--layer_quantile', type=str, default='q1')
parser.add_argument('--total_steps', type=int, default=100000)
args = parser.parse_args()

path = f'outputs/sae/{args.sae}/{args.lm_name}_{args.lm_size}_{args.split}_{args.lang}_{args.layer_quantile}'
os.makedirs(path, exist_ok=True)

activations = pickle.load(open(f'outputs/activations_{args.lm_name}_{args.lm_size}_{args.split}_{args.lang}.pkl', 'rb'))
    
dataset = CustomDataset(activations[f'{args.lang}_residual_{args.layer_quantile}'])

infos = {}
log_step = 100
visualize_step = 100
save_step = 10000
warmup_steps = 1000
total_steps = args.total_steps
activation_dim, dict_size = 4096, 100000
device='cuda:0'
lr=1e-4
l1_penalty=5e-2
resample_steps=10000
anneal_start=10
anneal_end=args.total_steps
n_sparsity_updates=100
sparsity_queue_length=10
p_start=1
p_end=0.1

if args.sae == 'standard':
    trainer = StandardTrainer(
        activation_dim=activation_dim,
        dict_size=dict_size,
        lr=lr, 
        l1_penalty=l1_penalty,
        warmup_steps=warmup_steps, # lr warmup period at start of training and after each resample
        device=device,
        resample_steps=resample_steps, # how often to resample neurons
    )
elif args.sae == 'gated':
    trainer = GatedTrainer(
        activation_dim=activation_dim,
        dict_size=dict_size,
        lr=lr, 
        warmup_steps=warmup_steps, # lr warmup period at start of training and after each resample
        device=device,
        resample_steps=resample_steps, # how often to resample neurons
        steps=total_steps
    )
elif args.sae == 'gated_anneal':
    trainer = GatedAnnealTrainer(
        activation_dim=activation_dim,
        dict_size=dict_size,
        lr=lr, 
        warmup_steps=warmup_steps, # lr warmup period at start of training and after each resample
        device=device,
        resample_steps=resample_steps, # how often to resample neurons
        steps=total_steps,
        anneal_start=anneal_start,
        anneal_end=anneal_end,
        n_sparsity_updates=n_sparsity_updates,
        sparsity_queue_length=sparsity_queue_length,
        p_end=p_end,
        p_start=p_start,
    )
elif args.sae == 'standard_anneal':
    trainer = StandardAnnealTrainer(
        activation_dim=activation_dim,
        dict_size=dict_size,
        lr=lr, 
        warmup_steps=warmup_steps, # lr warmup period at start of training and after each resample
        device=device,
        resample_steps=resample_steps, # how often to resample neurons
        p_start=p_start,
        p_end=p_end,
        anneal_start=anneal_start,
        anneal_end=anneal_end,
        n_sparsity_updates=n_sparsity_updates,
        sparsity_queue_length=sparsity_queue_length,
        steps=total_steps
    )   
else:
    raise ValueError(f"Invalid sae: {args.sae}")    

sns.set_style("whitegrid")
def visualize_infos(infos, figsize=(10,2.5)):
    steps = sorted(list(infos.keys()))
    names = list(infos[steps[0]].keys())
    fig, axes = plt.subplots(1, len(names), figsize=figsize)
    for i in range(len(names)):
        name = names[i]
        sns.lineplot(x=steps, y=[infos[step][name] for step in steps], ax=axes[i])
        axes[i].set_title(name)
        try:
            axes[i].set_ylim(0, np.quantile([infos[step][name] for step in steps], 0.80) )
        except:
            pass
    plt.tight_layout()
    return fig, axes 

import pickle 
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
pbar = tqdm(total=total_steps)
for step in range(total_steps):
    pbar.update(1)
    act = next(iter(dataloader))
    act = act.to(device)
    trainer.update(step, act)
    if step % log_step == 0:
        with torch.no_grad():
            infos[step] = trainer.loss(act, step, logging=True)[3]
            
    if step > 0 and step % visualize_step == 0:
        visualize_infos(infos, figsize=(12,3))
        plt.savefig(f"{path}/loss.png")
        plt.close()
        
    if step % save_step == 0:
        torch.save(trainer.ae.state_dict(), f"{path}/model.pth")
        torch.save(trainer.ae.state_dict(), f"{path}/model_{step}.pth")
        pickle.dump(infos, open(f"{path}/infos.pkl", "wb"))
        
        # check load 
        model_path =  f"{path}/model.pth"
        if args.sae == 'standard':
            loaded_ae = AutoEncoder.from_pretrained(model_path)
        elif args.sae == 'gated':
            loaded_ae = GatedAutoEncoder.from_pretrained(model_path)
        elif args.sae == 'gated_anneal':
            loaded_ae = GatedAutoEncoder.from_pretrained(model_path)
        elif args.sae == 'standard_anneal':
            loaded_ae = AutoEncoder.from_pretrained(model_path)
        del loaded_ae
        torch.cuda.empty_cache()
