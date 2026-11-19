"""Chronological training/validation split shared by preprocessing and training."""


def training_sample_count(train_size: int, validation_fraction: float) -> int:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")
    count = train_size - max(1, int(train_size * validation_fraction))
    if count < 1:
        raise ValueError("Not enough training samples after validation split.")
    return count
