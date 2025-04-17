import torch, inspect

def main():
    print("Torch version:", torch.__version__)
    sig = inspect.signature(torch.load)
    print("torch.load signature:", sig)
    print("Supports 'weights_only' parameter:", 'weights_only' in sig.parameters)

if __name__ == "__main__":
    main()
