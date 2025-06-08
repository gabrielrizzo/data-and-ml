import argparse
from text_classificator import use_nb_model

def main():
    parser = argparse.ArgumentParser(description='Classify text into categories')
    parser.add_argument('--text', type=str, help='The text to classify')
    args = parser.parse_args()
    
    if args.text is None:
        text = input("Please enter the text to classify: ")
    else:
        text = args.text
    
    use_nb_model(text)

if __name__ == '__main__':
    main()