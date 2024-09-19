from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import time
import inspect
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
#------------------------------------------------------

class CausalSelfAttention(nn.Module):

    #768 = 2^8 * 3
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd) # (input layer size, output layer size)
        #output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        #regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        #not really a 'bias', more of a mask, but following the OpenAI/HF naming though
        # under trigonal matrix를 만들기.
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1,1,config.block_size, config.block_size))
    
    def forward(self, x):
        B, T, C = x.size() #batch size, sequence length, embedding dimensionality (n_embd)
        #calculate query, key, values for all heads in batch and move head forward to be the batch
        #nh is "number of heads", hs is "head size", and C(number of channels) = nh * hs
        # e.g. in GPT-2 (124M). n_head = 12, hs = 64, so C = 768 channels in the Transformer
        time_qkv_0 = time.time()
        qkv = self.c_attn(x)
        time_qkv_1 = time.time()
        print(f"c_attn layer: {time_qkv_1-time_qkv_0} sec")
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # attention (materializes the large (T,T) matrix for all the queries and keys)
        
        #att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        #bias: future information:0, present information:1. 0 to -inf. 
        #att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf')) #autoregressive mask
        #att = F.softmax(att, dim=-1) # always 1 when added all outputs
        #y = att @ v #(B, nh, T, T) * (B, nh, T, hs) -> (B, nh, T, hs)
        
        # FlashAttention (굉장히 빨라짐!)
        time_flash_0 = time.time()
        y = F.scaled_dot_product_attention(q,k,v, is_causal=True)
        time_flash_1 = time.time()
        print(f"Flash Attention:{time_flash_1-time_flash_0} sec")

        y = y.transpose(1,2).contiguous().view(B,T,C) #re-assemble all head outputs side size 
        #output projection
        time_c_proj_0 = time.time()
        y = self.c_proj(y)
        time_c_proj_1 = time.time()
        print(f"c_proj layer {time_c_proj_1-time_c_proj_0} sec")
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
    def forward(self,x):
        time_cfc_0 = time.time()
        x = self.c_fc(x)
        time_cfc_1 = time.time()
        print(f"c_fc layer{time_cfc_1-time_cfc_0} sec")

        time_gelu_0 = time.time()
        x = self.gelu(x)
        time_gelu_1 = time.time()
        print(f"gelu calculated {time_gelu_1-time_gelu_0} sec")
        
        time_c_proj_0 = time.time()
        x = self.c_proj(x)
        time_c_proj_1 = time.time()
        print(f"c_proj layer {time_c_proj_1-time_c_proj_0} sec")
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        time_forward1_0 = time.time()
        x = x + self.attn(self.ln_1(x))
        time_forward1_1 = time.time()
        print(f"layer norm, attention, skip connection done. {time_forward1_1-time_forward1_0} sec")
        
        time_forward2_0 = time.time()
        x = x + self.mlp(self.ln_2(x))
        time_forward2_1 = time.time()
        print(f"layer norm, MLP, skip connection done. {time_forward2_1-time_forward2_0} sec")
        return x




@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),  
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight


        #init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std = std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        #idx is of shape (B,T), idx is token indices
        B, T = idx.size()
        #  T has to be less than block_size
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posistion embeddings
        pos = torch.arange(0,T,dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        #forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
    
    def configure_optimizers(self, weight_decay, learning_rate, device):
        # start with all of the candidate parameters( that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >=2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} params")
        print(f"num non=decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} params")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9,0.95), eps=1e-8, fused = use_fused)
        return optimizer
    
    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        #state dict.
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    
#----------------------------------------------------------------------
    
    




#----------------------------------------------------------------------
import tiktoken
import numpy as np
def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32) # added after video
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B  # batch size
        self.T = T  # sequence length
        self.process_rank = process_rank  # rank of the current process
        self.num_processes = num_processes  # total number of processes
        assert split in {'train', 'val'}, "split must be 'train' or 'val'"
        self.split = split

        # Load the data from 'input.txt'
        data_file = 'input.txt'
        with open(data_file, 'r', encoding='utf-8') as f:
            data = f.read()

        # Create a character-level vocabulary
        self.vocab = sorted(set(data))
        self.vocab_size = len(self.vocab)
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}  # char to index
        self.itos = {i: ch for i, ch in enumerate(self.vocab)}  # index to char

        # Encode the entire dataset
        tokens = [self.stoi[c] for c in data]

        # Split the data into training and validation sets (90% train, 10% val)
        split_idx = int(0.9 * len(tokens))
        if split == 'train':
            self.tokens = tokens[:split_idx]
        else:
            self.tokens = tokens[split_idx:]

        # Convert tokens to a PyTorch tensor
        self.tokens = torch.tensor(self.tokens, dtype=torch.long)

        # Initialize the data loader state
        self.reset()

    def reset(self):
        # Start position for the current process
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T

        # Check if there are enough tokens left for the next batch across all processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            # If not, reset to the beginning
            self.reset()

        # Slice out the buffer for the current batch
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = buf[:-1].view(B, T)  # input sequences
        y = buf[1:].view(B, T)   # target sequences

        # Advance the position for the next batch
        self.current_position += B * T * self.num_processes

        return x, y