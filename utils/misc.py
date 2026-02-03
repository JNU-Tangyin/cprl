def print_args(args):
    """
    Pretty-print argparse.Namespace or dict-like args.
    Used for experiment logging and debugging.
    """
    print("=" * 60)
    print("Experiment arguments:")
    print("-" * 60)

    if hasattr(args, "__dict__"):
        items = vars(args).items()
    elif isinstance(args, dict):
        items = args.items()
    else:
        raise TypeError("print_args expects argparse.Namespace or dict")

    for k, v in sorted(items):
        print(f"{k:<30}: {v}")

    print("=" * 60)
