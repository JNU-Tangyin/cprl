import argparse
from exp.exp_conformal import run_conformal_experiment

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, default="exchange_rate")
    parser.add_argument("--root_path", type=str, default="./dataset/exchange_rate")
    parser.add_argument("--data_path", type=str, default="exchange_rate.csv")
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--label_len", type=int, default=48)
    parser.add_argument("--pred_len", type=int, default=24)
    parser.add_argument("--features", type=str, default="S")
    parser.add_argument("--target", type=str, default="0T")
    parser.add_argument("--freq", type=str, default="d")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--comment", type=str, default="baseline")

    args = parser.parse_args()
    run_conformal_experiment(args)
