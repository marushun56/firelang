import argparse
import os
import torch
from firelang import FireWord
from firelang.utils.log import logger

def main():
    parser = argparse.ArgumentParser(description="Interactive word similarity demo.")
    parser.add_argument("--model_path", type=str, default="results/0PZW2SR3/best", help="Path to the trained model directory.")
    parser.add_argument("--words", type=str, nargs='+', help="List of words to check similarity for. If not provided, interactive mode is used.")
    parser.add_argument("--k", type=int, default=10, help="Number of similar words to return.")
    parser.add_argument("--cpu", action="store_true", help="Use cpu rather than CUDA.")
    
    args = parser.parse_args()
    
    device = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {args.model_path}...")
    try:
        model = FireWord.from_pretrained(args.model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
            
    model = model.to(device)
    model.eval()
    print("Model loaded successfully.")

    if args.words:
        for word in args.words:
            try:
                similar_words = model.most_similar(word, k=args.k)
                print(f"\nTop {args.k} similar words to '{word}':")
                for w, score in similar_words:
                    print(f"  {w}: {score:.4f}")
            except KeyError:
                print(f"Word '{word}' not found in vocabulary.")
            except Exception as e:
                print(f"Error calculating similarity: {e}")
        return

    while True:
        try:
            word = input("\nEnter a word (or 'q' to quit): ").strip()
            if word.lower() in ['q', 'quit', 'exit']:
                break
            
            if not word:
                continue

            try:
                similar_words = model.most_similar(word, k=args.k)
                print(f"\nTop {args.k} similar words to '{word}':")
                for w, score in similar_words:
                    print(f"  {w}: {score:.4f}")
            except KeyError:
                print(f"Word '{word}' not found in vocabulary.")
            except Exception as e:
                print(f"Error calculating similarity: {e}")

        except KeyboardInterrupt:
            break
    
    print("\nExiting...")

if __name__ == "__main__":
    main()
