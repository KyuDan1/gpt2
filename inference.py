from train_gpt2 import *
print("model instance start")
model = GPT(GPTConfig(vocab_size=50304))
print("made GPT structure")
print("model load start")
model.load_state_dict(torch.load('mygpt2model.pth'))
print("loaded_state_dict")
model.to('cpu')
print("model to cpu")
#prefix tokens
model.eval()
print("model eval")


num_return_sequences = 5
import tiktoken
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I am Tim, and")
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
x = tokens.to('cpu')
print("input goes to cpu")

max_length = 100 

# generate! right now x is (B, T) where B = 5, T = 8
# Set the seed to 42
torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    # forward the model to get the logits
    with torch.no_grad():
        time_model_0 = time.time()
        logits = model(x) # (B, T, vocab_size)
        time_model_1 = time.time()
        print(f"logit calculate done {time_model_1-time_model_0} sec")
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