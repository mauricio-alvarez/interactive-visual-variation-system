import torch


def main() -> None:
    print(f"torch: {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu: {torch.cuda.get_device_name(0)}")
        print(f"cuda runtime: {torch.version.cuda}")


if __name__ == "__main__":
    main()

