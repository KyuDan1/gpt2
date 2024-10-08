from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
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
        
        x0 = x
        time_ln_1_0 = time.time()
        x = self.ln_1(x)
        time_ln_1_1 = time.time()
        print(f"layer norm 1 {time_ln_1_1-time_ln_1_0} sec")
        time_attn_0 = time.time()
        x = self.attn(x)
        time_attn_1 = time.time()
        print(f"attn_1 {time_attn_1-time_attn_0} sec")
        time_skip_0 = time.time()
        x = x0 + x
        time_skip_1 = time.time()
        print(f"skip connection {time_skip_1-time_skip_0} sec")
        

        x1 = x
        time_ln_1_0 = time.time()
        x = self.ln_2(x)
        time_ln_1_1 = time.time()
        print(f"layer norm 2 {time_ln_1_1-time_ln_1_0} sec")
        time_attn_0 = time.time()
        x = self.mlp(x)
        time_attn_1 = time.time()
        print(f"mlp {time_attn_1-time_attn_0} sec")
        time_skip_0 = time.time()
        x = x1 + x
        time_skip_1 = time.time()
        print(f"skip connection 2{time_skip_1-time_skip_0} sec")

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


#----------------------------------------------------------------------
import time
import os

# simple launch:
# python train_gpt2.py
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=8 train_gpt2.py


from torch.distributed import init_process_group, destroy_process_group

#set up DDP (distributed data parallel).
#torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 #this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    #apple silicon mps
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")


torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)


total_batch_size = 524288 # 2**19, ~0.5M, in number of tokens
B = 4 # micro batch size
T = 1024 # sequence length
assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B=B, T=T, process_rank = ddp_rank, num_processes=ddp_world_size, split="train")
val_loader = DataLoaderLite(B=B, T=T, process_rank = ddp_rank, num_processes=ddp_world_size, split="val")

torch.set_float32_matmul_precision('high')

# import tiktoken
"""enc = tiktoken.get_encoding('gpt2')
with open('input.txt', 'r') as f:
    text = f.read()
text = text[:1000]
tokens = enc.encode(text)
B, T = 4, 32
buf = torch.tensor(tokens[:B*T + 1])
buf = buf.to(device)
x = buf[:-1].view(B,T)
y = buf[1:].view(B,T)"""

# create model
model = GPT(GPTConfig(vocab_size=50304)) # 50257이 기본인데, 2의 제곱이 많이 들어있는 50304로 바꿈. More FLOPS BUT 조금 더 빨라짐.
model.to(device)
use_compile = False # torch.compile interferes with HellaSwag eval and Generation. TODO fix
if use_compile:
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model # always contains the "raw" unwrapped model


max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr* (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)



#optimizer
#optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)

optimizer = raw_model.configure_optimizers(weight_decay = 0.1, learning_rate=6e-4, device = device)


for step in range(max_steps):
    t0 = time.time()
    # Once in a while, evaluate our validation loss
    if step % 10 == 0:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                val_loss_accum += loss.item()
            # Compute average loss across validation steps
            val_loss_accum /= val_loss_steps
            # Convert to tensor for reduction
            val_loss_tensor = torch.tensor(val_loss_accum, device=device)
        if ddp:
            # Reduce across all processes to get the sum, then average
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
            val_loss_tensor /= dist.get_world_size()
            val_loss_accum = val_loss_tensor.item()
        if master_process:
            print(f"Validation loss: {val_loss_accum:.4f}")


    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x,y) # mixed precision - model의 weight는 fp32로, logit이나 loss는 bf16으로..
        loss = loss / grad_accum_steps
        #import code; code.interact(local=locals()) # torch의 기본 데이터 타입은 torch.float32 다. 32bit를 써서 실수를 구분함.
        # FP32보다 FP16이 16배 계산량이 많다. (빠르다.)
        # INT8은 train할 땐 안 쓰고 inference할 때만 쓴다.
        loss_accum += loss.detach()
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps -1)
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # parameter의 norm이 1보다 크면 잘라냄.
                                                                   # norm이 너무 크다는 것은 너무 큰 loss를 얻는 다는건데, model이 shock을 먹고 훈련이 안정되지 않을 수 있음.
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1-t0) # time difference in milliseconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    if master_process:
        print(f"step {step} | loss: {loss_accum.item():.6f} | lr: {lr} | norm: {norm:.4f} | dt:{(dt*100):.2f}ms | tok/sec: {tokens_per_sec:.2f}")

# Save the model (only on the master process)
if master_process:
    torch.save(model.state_dict(), 'mygpt2model.pth')

if ddp:
    destroy_process_group()



#import sys; sys.exit(0)

model = GPT(GPTConfig(vocab_size=50304))
model.load_state_dict(torch.load('mygpt2model.pth'))
model.to('cuda')
#prefix tokens
model.eval()


num_return_sequences = 5
import tiktoken
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I am Tim, and")
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
x = tokens.to('cuda')


max_length = 100 

# generate! right now x is (B, T) where B = 5, T = 8
# Set the seed to 42
torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    # forward the model to get the logits
    with torch.no_grad():
        logits = model(x) # (B, T, vocab_size)
        # take the logits at the last position
        logits = logits[:,-1,:] #(B, vocab_size)
        # get the probabilities
        probs = F.softmax(logits, dim=-1)
        # do top-k sampling of 50 (huggingface pipeline default)
        # topk_probs here becomes (5, 50), topk_indices is (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # select a token from the top-k probabilities
        ix = torch.multinomial(topk_probs, 1) # (B, 1)
        xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
        # append to the sequence
        x = torch.cat((x, xcol), dim=1)

# print the generated text
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)