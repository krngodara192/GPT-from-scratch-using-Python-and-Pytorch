# Libraries to import
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import argparse

parser = argparse.ArgumentParser(description = 'This is a custom gpt')
#Here we add an argument to parser, specfying the expected type, a help message, etc.
parser.add_argument('-batch_size', type=int, required=True, help='Please provide batch size')
args = parser.parse_args()
#Now we can use the argument in our program
print(f'batch size: {args.batch_size}')

device = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters
block_size = 64  # Numbr of tokens to consider in 1 block
batch_size = args.batch_size  # Number of parallerl blocks
max_iters = 3000  # Number of epochs to train
eval_iters = 100  # After how many epochs loss to be reported
eval_interval = 500
learning_rate = 3e-4  # Decides how fast we need to converge model
n_embd = 384
n_layer = 4
n_head = 4
dropout = 0.2

# Open the text file and save all characters and vocabulary size
chars = ""
with open("D:\LLM\Documents\wizard_of_oz.txt", 'r', encoding= 'utf-8') as f:
    text = f.read()
    chars = sorted(list(set(text)))

vocabulary_size = len(chars)

# Create the tokenizer to encode and decode the text
string_to_int = { ch:i for i,ch in enumerate(chars)} # create mapping for token and embed
int_to_string = { i:ch for i,ch in enumerate(chars)}
encode = lambda s: [string_to_int[c] for c in s]
decode = lambda s: ''.join([int_to_string[c] for c in s])


class Head(nn.Module):
    """ one head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input size - (batch, time-step, channnels)
        # output size - (batch, time-step, head_size)
        B,T,C = x.shape
        k = self.key(x) # (B,T,hs)
        q = self.query(x) # (B,T,hs)

        # Compute attantion scores
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B,T,hs) @ (B,hs,T) -> (B,T,C)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B,T,T)
        wei = F.softmax(wei, dim=-1) # (B,T,T)

        # Perform the weighted aggregation
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B,T,T) @ (B,T,hs) -> (B,T,hs)
        return out

class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size*num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

# Create the feed forward layer in decoder block
class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

# Create the decoder block as mentioned in transformer paper
class Block(nn.Module):

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd//n_head
        self.sa = MultiHeadAttention(n_head, head_size)  # Self attention layer
        self.ffwd = FeedForward(n_embd) # Feed Forward layer
        self.ln1 = nn.LayerNorm(n_embd) # Norm layer
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        y = self.sa(x)
        x = self.ln1(x+y)
        y = self.ffwd(x)
        x = self.ln2(x+y)
        return x

# create the GPT model
class GPTLanguageModel(nn.Module): # NN.module is from pytorch - neural network module
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)  # Create the trainable embedding table
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)]) #create seq layers depending on no of decoders
        self.ln_f = nn.LayerNorm(n_embd) # create normalization layer
        self.lm_head = nn.Linear(n_embd, vocab_size) # create linear transformation layer
        self.apply(self.__init__weights)
    
    # Function to define intial weights for better convergence
    def __init__weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, index, targets=None):
        B,T = index.shape

        #idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(index) #(B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) #(T,C)
        x = tok_emb + pos_emb #(B,T,C)
        x = self.blocks(x) #(B,T,C)
        x = self.ln_f(x) #(B,T,C)
        logits = self.lm_head(x) #(B,T,vocab_size)

        if targets == None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, index, max_new_tokens):
        # index is (B,T) array of indices in current context
        for _ in range(max_new_tokens):
            #crop index to the last block size tokens
            index_cond = index[:,-block_size:]
            #get the predictions
            logits, loss = self.forward(index_cond)
            # focus only on last time step
            logits = logits[:,-1,:] # becomes (B,C)
            # get probablities using softmax
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            index_next = torch.multinomial(probs, num_samples=1) # (B,1)
            # append sampled text into running sequence
            index = torch.cat((index, index_next), dim=1) #(B,T+1)
        return index

model = GPTLanguageModel(vocabulary_size)
print('Loading model parameters...')
with open('model-01.pkl', 'rb') as f:
    model = pickle.load(f)
print('Model loaded succesfully')
m = model.to(device)  # load model to device - cpu or cuda

while True:
    prompt = input("Prompt\n")
    context = torch.tensor(encode(prompt), dtype=torch.long, device=device)
    #max new tokens cannot be above block size
    generated_chars = decode(m.generate(context.unsqueeze(0), max_new_tokens=150)[0].tolist())
    print(f'Completion:\n{generated_chars}')
