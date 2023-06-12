from lightning_trainable.hparams import AttributeDict

from .trainable import TrainableHParams


class GaussianizationHParams(TrainableHParams):
    inputs: int
    layers: int
    rotation: dict
    transform: dict
    end_to_end: bool
    loss_free: bool
    init_transform_with_fit: bool
    data_distribution: dict
    latent_distribution: dict = dict(kind="normal")
    train_samples: int = 60_000
    val_samples: int = 10_000
    test_samples: int = 10_000
    resample: bool = True
    epochs_per_layer: int

    # delete max epochs
    max_epochs: None | int = None

    # batch size is technically optional
    batch_size: int | None = None

    @classmethod
    def validate_parameters(cls, hparams: AttributeDict) -> AttributeDict:
        hparams = super().validate_parameters(hparams)

        if hparams["loss_free"]:
            hparams["batch_size"] = hparams["train_samples"]

            if hparams["epochs_per_layer"] > 1:
                raise ValueError(f"Loss-free training does not support multiple epochs per layer.")

        if hparams["loss_free"] and not hparams["init_transform_with_fit"]:
            raise ValueError("Layer-wise training without loss must be fit.")
        if hparams["end_to_end"] and hparams["init_transform_with_fit"]:
            raise ValueError("Only layer-wise training can initialize with pre-fit layers.")

        expected_max_epochs = hparams["layers"] * hparams["epochs_per_layer"]
        if hparams["max_epochs"] is None:
            hparams["max_epochs"] = expected_max_epochs
        else:
            if hparams["max_epochs"] != expected_max_epochs:
                raise ValueError(
                    f'max_epochs inconsistent with layers * epochs_per_layer: '
                    f'{hparams["max_epochs"]} != {expected_max_epochs}'
                )

        if hparams["end_to_end"] and hparams["loss_free"]:
            raise ValueError(f"Cannot train end-to-end and loss-free.")

        return hparams
