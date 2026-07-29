from robometrics.datasets import demo_raw, motion_benchmaker_raw, mpinets_raw


DATASETS = {
    "demo": demo_raw,
    "motion_benchmaker": motion_benchmaker_raw,
    "mpinets": mpinets_raw,
}


def load_problems(dataset: str = "demo"):
    loader = DATASETS.get(dataset)
    if loader is None:
        raise ValueError(
            f"Unknown dataset '{dataset}'. "
            f"Choose from: {list(DATASETS.keys())}"
        )
    return loader()
