# data_factory.py
from torch.utils.data import DataLoader, Subset
from data_loader import CSVTimeSeriesDataset  # 路径按你的项目调整

# 和原库统一风格：名字 → Dataset 类
data_dict = {
    "custom": CSVTimeSeriesDataset,
    # 以后你想加别的，比如 'm4': M4Dataset，也可以往这里扩展
}


def data_provider(args, flag):
    """
    和时间序列项目里的 data_provider 接口类似：
        data_set, data_loader = data_provider(args, flag)

    不同点：
        - Dataset 本身不再做 train/val/test 划分
        - 这里用 Subset 根据 flag 划分 train / calib(val) / test
    """
    assert flag in ["train", "val", "calib", "test", "TEST"]

    Data = data_dict[args.data]  # 例如 args.data = 'custom'
    timeenc = 0 if args.embed != "timeF" else 1

    # 1. 构建完整时间序列数据集（不划分）
    full_dataset = Data(
        args=args,
        root_path=args.root_path,
        data_path=args.data_path,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        scale=True,
        timeenc=timeenc,
        freq=args.freq,
    )

    N = len(full_dataset)
    # 可以通过 args 决定比例，没设置就走默认：0.6 / 0.2 / 0.2
    train_ratio = getattr(args, "train_ratio", 0.6)
    calib_ratio = getattr(args, "calib_ratio", 0.2)

    n_train = int(N * train_ratio)
    n_calib = int(N * calib_ratio)
    n_test = N - n_train - n_calib

    idx_train = range(0, n_train)
    idx_calib = range(n_train, n_train + n_calib)
    idx_test = range(n_train + n_calib, N)

    if flag == "train":
        indices = idx_train
    elif flag in ["val", "calib"]:
        indices = idx_calib
    else:  # 'test' or 'TEST'
        indices = idx_test

    subset = Subset(full_dataset, indices)

    shuffle_flag = True if flag == "train" else False
    drop_last = False
    batch_size = args.batch_size

    data_loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
    )

    print(flag, len(subset))
    return subset, data_loader
