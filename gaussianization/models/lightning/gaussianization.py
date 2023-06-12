from math import ceil
from typing import Optional, Union

import numpy as np
import torch
import torch.distributions as D

import FrEIA as freia
import FrEIA.framework as ff
import FrEIA.modules as fm


from gaussianization import utils
from gaussianization.distributions import *
from gaussianization.models import rotations, transforms
from .trainable import Trainable
from .gaussianization_hparams import GaussianizationHParams


class Gaussianization(Trainable):
    def __init__(self, hparams: GaussianizationHParams | dict, log_dir=None):
        if not isinstance(hparams, GaussianizationHParams):
            hparams = GaussianizationHParams(**hparams)

        super().__init__(hparams)

        self.train_data_distribution, self.test_data_distribution = self.configure_distribution(
            **self.hparams.data_distribution)
        self.latent_distribution, _ = self.configure_distribution(**self.hparams.latent_distribution)

        self.model = self.configure_model()

        self.log_dir = log_dir

        self.train_data, self.val_data, self.test_data = self.configure_datasets()

        self.epoch = 0

        self.pushforward = freia.distributions.PushForwardDistribution(
            self.train_data_distribution,
            self.model,
        )

        self.pullback = freia.distributions.PullBackDistribution(
            self.latent_distribution,
            self.model,
        )

    def forward(self, batch, batch_idx=None):
        if not torch.is_tensor(batch):
            # get the first tensor (e.g. when using TensorDataset)
            batch = batch[0]
            assert torch.is_tensor(batch)

        z, log_jac_det = self.model(batch)

        return z, log_jac_det

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        assert torch.is_tensor(x)
        return self.pullback.log_prob(x)

    def loss(self, batch, batch_idx=None) -> torch.Tensor:
        if not torch.is_tensor(batch):
            batch = batch[0]
            assert torch.is_tensor(batch)

        return -self.log_prob(batch).mean(dim=0)

    def training_step(self, batch, batch_idx):
        if not torch.is_tensor(batch):
            batch = batch[0]
            assert torch.is_tensor(batch)

        if self.hparams.end_to_end is True:
            # standard end-to-end lossy training
            return super().training_step(batch, batch_idx)

        _, rotation, transform = self.get_epoch_layers(self.epoch)

        if batch_idx == 0:
            # get the whole dataset
            data = self.train_data.tensor

            # fit the rotation
            rotation.fit(data)

            data, log_jac_det_rot = freia.utils.tuple_free_batch_forward(
                module=rotation,
                data=data,
                batch_size=1024,
                device=self.device,
                loader_kwargs=dict(num_workers=4)
            )

            # fit the transform
            if self.hparams.init_transform_with_fit:
                transform.fit(data)

        if self.hparams.loss_free is True:
            # already done
            assert batch_idx == 0, "Loss-free training does not support batching."
            return torch.tensor(0., requires_grad=True)

        (batch,), log_jac_det_rot = rotation((batch,))
        (z,), log_jac_det_transform = transform((batch,))
        log_prob = self.latent_distribution.log_prob(z)

        loss = -(log_prob + log_jac_det_rot + log_jac_det_transform).mean(dim=0)

        self.log("training_loss", loss)

        return loss

    def on_train_epoch_end(self):
        self.resample()

        self.epoch += 1

    def configure_optimizers(self):
        if self.hparams.loss_free is True:
            return None

        return super().configure_optimizers()

    def configure_model(self):
        self.model = ff.SequenceINN(self.hparams.inputs)

        for i in range(self.hparams.layers):
            self.add_rotation()
            self.add_transform()

        return self.model

    def add_rotation(self):
        match self.hparams.rotation.get("kind").lower():
            case "ica":
                rotation = rotations.ICARotation
            case "max-kswd" | "max_kswd":
                rotation = rotations.MaxKSWDRotation
            case "pca":
                rotation = rotations.PCARotation
            case "random":
                rotation = rotations.RandomRotation
            case "none":
                rotation = rotations.NoRotation
            case rotation:
                raise NotImplementedError(f"Unsupported rotation: {rotation}")

        rotation_kwargs = self.hparams.rotation.copy()
        rotation_kwargs.pop("kind")

        self.model.append(rotation, **rotation_kwargs)

        return self.model[-1]

    def add_transform(self):
        match self.hparams.transform.get("kind").lower():
            case "spline":
                transform = transforms.RQSpline
            case "stable-spline":
                transform = transforms.Spline1d
            case "affine-coupling":
                transform = fm.GLOWCouplingBlock
            case "spline-coupling":
                transform = fm.RationalQuadraticSpline
            case None:
                raise ValueError("You must supply a transform kind.")
            case transform:
                raise NotImplementedError(f"Unsupported transform: {transform}")

        transform_kwargs = self.hparams.transform.copy()
        kind = transform_kwargs.pop("kind")

        if "coupling" in kind:
            widths = transform_kwargs.pop("subnet_widths")
            activation = transform_kwargs.pop("activation")
            dropout = transform_kwargs.pop("dropout")

            def subnet_constructor(dims_in, dims_out):
                subnet = utils.make_dense([dims_in, *widths, dims_out], activation=activation, dropout=dropout)
                subnet[-1].weight.data.zero_()
                subnet[-1].bias.data.zero_()

                return subnet

            transform_kwargs["subnet_constructor"] = subnet_constructor

        self.model.append(transform, **transform_kwargs)

        return self.model[-1]

    def resample(self):
        # resample
        if self.hparams.resample is True:
            self.train_data.resample()

        # push data
        if self.hparams.end_to_end is False:
            prev_model, rotation, transform = self.get_epoch_layers(self.epoch)

            if self.hparams.resample is True:
                self.train_data.push_data(prev_model, device=self.device)

            self.train_data.push_data(rotation, device=self.device)
            self.train_data.push_data(transform, device=self.device)

    def configure_datasets(self):
        train_data = SampleDataset(self.train_data_distribution, self.hparams.train_samples)
        val_data = SampleDataset(self.test_data_distribution, self.hparams.val_samples)
        test_data = SampleDataset(self.test_data_distribution, self.hparams.test_samples)

        return train_data, val_data, test_data

    def configure_distribution(self, **kwargs):
        kind = kwargs.pop("kind")

        match kind.lower():
            case "gaussian" | "normal":
                inputs = self.hparams.inputs
                loc = torch.zeros(inputs)
                scale = torch.ones(inputs)
                test_distribution = train_distribution = D.Independent(D.Normal(loc, scale), 1)

            case "gaussian_mixture":
                mixtures = kwargs["mixtures"]
                inputs = self.hparams.inputs
                logits = torch.zeros(mixtures)

                match kwargs.get("means"):
                    case "random":
                        means = 10.0 * torch.from_numpy(
                            np.random.default_rng(1892).normal(size=(mixtures, inputs))
                        ).float()
                    case torch.Tensor() as means:
                        pass
                    case None:
                        raise ValueError(f"You must supply mixture means.")
                    case means:
                        raise NotImplementedError(f"Unsupported mixture means: {means}")

                match kwargs.get("stds"):
                    case "random":
                        stds = 1.0 + torch.from_numpy(
                            np.random.default_rng(128).uniform(size=(mixtures, inputs))
                        ).float()
                    case torch.Tensor() as stds:
                        pass
                    case None:
                        raise ValueError(f"You must supply mixture stds.")
                    case stds:
                        raise NotImplementedError(f"Unsupported mixture stds: {stds}")

                test_distribution = train_distribution = D.MixtureSameFamily(
                    mixture_distribution=D.Categorical(logits=logits),
                    component_distribution=D.Independent(D.Normal(loc=means, scale=stds), 1),
                )

            case "cov_gaussian":
                test_distribution = train_distribution = MultivariateGaussianDistribution(self.hparams.inputs, **kwargs)

            case "banana":
                test_distribution = train_distribution = BananaDistribution(self.hparams.inputs, **kwargs)

            case "banana-tree":
                test_distribution = train_distribution = BananaTreeDistribution(self.hparams.inputs, **kwargs)

            case "emnist-multiscale":
                if len(kwargs) > 0:
                    raise ValueError(f"Unnecessary distribution params: {kwargs!r}")
                train_distribution = EMNISTMultiscaleDistribution(self.hparams.inputs, train=True)
                test_distribution = EMNISTMultiscaleDistribution(self.hparams.inputs, train=False)

            case "emnist-model":
                test_distribution = train_distribution = EMNISTModelDistribution(self.hparams.inputs, **kwargs)

            case "fashion-mnist-model":
                test_distribution = train_distribution = FashionMNISTModelDistribution(self.hparams.inputs, **kwargs)

            case "cat-faces-model":
                test_distribution = train_distribution = CatFacesModelDistribution(self.hparams.inputs, **kwargs)

            case None:
                raise ValueError(f"You must support a distribution kind.")

            case distribution:
                raise NotImplementedError(f"Unsupported distribution: {distribution}")

        # pre-processing transform
        pre_transform = ff.SequenceINN(self.hparams.inputs)
        pre_transform.append(
            fm.ActNorm,
            init_data=train_distribution.sample((self.hparams.train_samples,))
        )

        train_distribution = freia.distributions.PushForwardDistribution(train_distribution, transform=pre_transform)
        test_distribution = freia.distributions.PushForwardDistribution(test_distribution, transform=pre_transform)

        return train_distribution, test_distribution

    @torch.enable_grad()
    def fit(self, **extra_kwargs):
        self.epoch = 0
        device = self.device
        trainer = self.configure_trainer(self.log_dir, **extra_kwargs)

        trainer.fit(self)
        validation_loss = trainer.validate(self)[0]["validation_loss"]

        # keep model on gpu after training
        self.to(device)

        return validation_loss

    def sample(self, sample_shape: torch.Size = (1,)) -> torch.Tensor:
        return self.pullback.sample(sample_shape)

    def to(self, *args, **kwargs):
        self.train_data_distribution.force_to(*args, **kwargs)
        self.test_data_distribution.force_to(*args, **kwargs)
        self.latent_distribution.force_to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def cuda(self, device: Optional[Union[torch.device, int]] = None):
        # Delegate to .to(...) as it moves distributions, too
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        elif isinstance(device, int):
            device = torch.device("cuda", index=device)
        return self.to(device)

    def cpu(self):
        # Delegate to .to(...) as it moves distributions, too
        return self.to("cpu")

    def sample_data(self, *args, train=True, **kwargs):
        return (
            self.train_data_distribution
            if train else
            self.test_data_distribution
        ).sample(*args, **kwargs).to(self.device)

    def sample_latent(self, *args, **kwargs):
        return self.latent_distribution.sample(*args, **kwargs).to(self.device)

    def get_epoch_layers(self, epoch: int):
        logical_layer = epoch // self.hparams.epochs_per_layer
        prev_layers = self.model[:2 * logical_layer]
        rotation = self.model[2 * logical_layer]
        transform = self.model[2 * logical_layer + 1]

        return prev_layers, rotation, transform
