"""Wire the training CLI, application use case, and ATLOP adapter."""
from application.training import run_training
from interfaces.train import parse_args


def main():
    args = parse_args()
    from infrastructure.training import ATLOPTraining

    metrics = run_training(ATLOPTraining(args), checkpoint=args.load_path)
    if metrics is not None:
        print(metrics)


if __name__ == "__main__":
    main()
