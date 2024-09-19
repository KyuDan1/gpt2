from temp_classes import GPT, GPTConfig
import torch
import time
from torch.nn import functional as F
import time
import torch
from torch.nn import functional as F
import tiktoken
import sys

# 파일에 출력 내용을 저장하기 위해 sys.stdout을 파일로 변경
with open("generated_text_output.txt", "w") as f:
    # 표준 출력을 파일로 변경
    sys.stdout = f

    device = "cpu"
    """if torch.cuda.is_available():
        device = "cuda"
    # Apple silicon MPS
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"""
    print(f"using device: {device}")

    print("model instance start")
    model = GPT(GPTConfig(vocab_size=50304))
    print("made GPT structure")
    print("model load start")
    model.load_state_dict(torch.load('mygpt2model.pth', map_location=torch.device(device)))
    print("loaded_state_dict")
    model.to(device)
    print("model to device")
    # prefix tokens
    model.eval()
    print("model eval")

    num_return_sequences = 5

    enc = tiktoken.get_encoding('gpt2')
    tokens = enc.encode("Hello, I am Tim, and")
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    x = tokens.to(device)
    print("input goes to device")

    max_length = 100

    # generate! right now x is (B, T) where B = 5, T = 8
    # Set the seed to 42
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    while x.size(1) < max_length:
        # forward the model to get the logits
        with torch.no_grad():
            time_model_0 = time.time()

            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(x)  # (B, T, vocab_size)

            time_model_1 = time.time()
            print(f"logit calculate done: {time_model_1-time_model_0} sec")

            # take the logits at the last position
            logits = logits[:, -1, :]  # (B, vocab_size)
            # get the probabilities
            probs = F.softmax(logits, dim=-1)
            # do top-k sampling of 50 (huggingface pipeline default)
            # topk_probs here becomes (5, 50), topk_indices is (5, 50)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            # select a token from the top-k probabilities
            ix = torch.multinomial(topk_probs, 1)  # (B, 1)
            xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
            # append to the sequence
            x = torch.cat((x, xcol), dim=1)

    # print the generated text
    for i in range(num_return_sequences):
        tokens = x[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        print(">", decoded)

# 다시 표준 출력으로 복원
sys.stdout = sys.__stdout__
