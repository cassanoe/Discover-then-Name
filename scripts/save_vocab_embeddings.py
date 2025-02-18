import sys
import torch
import clip
from pathlib import Path

def load_words(file_path):
    with open(file_path, 'r') as file:
        words = [line.strip() for line in file.readlines()]
    return words

def save_embeddings(embeddings, save_path):
    torch.save(embeddings, save_path)

def main(clip_model_name, words_file, save_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load the CLIP model
    model, preprocess = clip.load(clip_model_name, device=device)
    
    # Load words from the text file
    words = load_words(words_file)
    
    # Tokenize the words
    text_inputs = clip.tokenize(words).to(device)
    
    # Generate embeddings
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
    
    # Save the embeddings
    save_embeddings(text_features, save_path)
    print(f"Embeddings saved to {save_path}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python save_embeddings.py <clip_model_name> <words_file> <save_path>")
        sys.exit(1)
    
    clip_model_name = sys.argv[1]
    words_file = sys.argv[2]
    save_path = sys.argv[3]
    
    main(clip_model_name, words_file, save_path)