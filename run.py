import sys
from transpiler import run_card_file

def main():
    if len(sys.argv) < 2:
        print("Usage: python run.py <filename.card>")
        return

    file_path = sys.argv[1]
    output, error = run_card_file(file_path)

    if error:
        print(f"Error during execution: {error}")
    else:
        print(output)

if __name__ == "__main__":
    main()