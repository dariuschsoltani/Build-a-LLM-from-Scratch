# Build-a-LLM-from-Scratch
Implementation of Build a Large Language Model from Scratch from Sebastian Raschka

Chapter 2: 
1. Raw text is broken into tokens: Each word and punctuation mark corresponds to a token. Unknown words and end of sentences are marked with special tokens <|unk|> and <|endoftext|>.
2. Tokens are converted into Token ID's: We built a simple Encoder by sorting the tokens lexicographically and mapping them to 0,1,2, etc. But we used the BPE (BPE Tokenizer), the one which gpt2 uses, which is more sophisticated. It handles unknown words by splitting them into subwords (which are tokenized) and it builds its vocabulary automatically (instead of statically as our simple Encoder) by merging frequent pairs learned from data. 
3. Token ID's are mapped to k-dimensional token embedding vectors which capture the semantics of the token. 
   A k-dimensional positional vector (absolute or relative) is added to the token embedding vectors to capture positional
   information of a token. 
   At first, the token embedding vectors contain random numbers which will be changed during the model training via backpropagation. 
4. We have an input torch which stores the input embeddings and a targets torch which stores the target embeddings:
   Batch size: number of input/target vectors in a batch. 
   
