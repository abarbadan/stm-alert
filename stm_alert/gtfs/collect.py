import argparse
from stm_alert.gtfs.utils import load_config
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()

def main():
    

if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    print(config)
    main()