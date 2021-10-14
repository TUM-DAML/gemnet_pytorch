import numpy as np
import logging
import torch

from .schedules import LinearWarmupExponentialDecay
from .ema_decay import ExponentialMovingAverage


class Trainer:
    """
    Parameters
    ----------
        model: Model
            Model to train.
        learning_rate: float
            Initial learning rate.
        decay_steps: float
            Number of steps until learning rate reaches learning_rate*decay_rate
        decay_rate: float
            Decay rate.
        warmup_steps: int
            Total number of warmup steps of the learning rate schedule..
        weight_decay: bool
            Weight decay factor of the AdamW optimizer.
        staircase: bool
            If True use staircase decay and not (continous) exponential decay
        grad_clip_max: float
            Gradient clipping threshold.
        decay_patience: int
            Learning rate decay on plateau. Number of evaluation intervals after decaying the learning rate.
        decay_factor: float
            Learning rate decay on plateau. Multiply inverse of decay factor by learning rate to obtain new learning rate.
        decay_cooldown: int
            Learning rate decay on plateau. Number of evaluation intervals after which to return to normal operation.
        ema_decay: float
            Decay to use to maintain the moving averages of trained variables.
        rho_force: float
            Weighing factor for the force loss compared to the energy. In range [0,1]
            loss = loss_energy * (1-rho_force) + loss_force * rho_force
        loss: str
            Name of the loss objective of the forces.
        mve: bool
            If True perform Mean Variance Estimation.
        agc: bool
            If True use adaptive gradient clipping else clip by global norm.
    """

    def __init__(
        self,
        model,
        learning_rate: float = 1e-3,
        decay_steps: int = 100000,
        decay_rate: float = 0.96,
        warmup_steps: int = 0,
        weight_decay: float = 0.001,
        staircase: bool = False,
        grad_clip_max: float = 1000,
        decay_patience: int = 10,  # decay lr on plateau by decay_factor
        decay_factor: float = 0.5,
        decay_cooldown: int = 10,
        ema_decay: float = 0.999,
        rho_force: float = 0.99,
        loss: str = "mae",  # else use rmse
        mve: bool = False,
        agc=False,
    ):
        assert 0 <= rho_force <= 1

        self.model = model
        self.ema_decay = ema_decay
        self.grad_clip_max = grad_clip_max
        self.rho_force = float(rho_force)
        self.mve = mve
        self.loss = loss
        self.agc = agc

        if mve:
            self.tracked_metrics = [
                "loss",
                "energy_mae",
                "energy_nll",
                "energy_var",
                "force_mae",
                "force_rmse",
                "force_nll",
                "force_var",
            ]
        else:
            self.tracked_metrics = ["loss", "energy_mae", "force_mae", "force_rmse"]

        self.reset_optimizer(
            learning_rate,
            weight_decay,
            warmup_steps,
            decay_steps,
            decay_rate,
            staircase,
            decay_patience,
            decay_factor,
            decay_cooldown,
        )

    def reset_optimizer(
        self,
        learning_rate,
        weight_decay,
        warmup_steps,
        decay_steps,
        decay_rate,
        staircase,
        decay_patience,
        decay_factor,
        decay_cooldown,
    ):
        if weight_decay > 0:
            adamW_params = []
            rest_params = []
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    if "atom_emb" in name:
                        rest_params += [param]
                        continue
                    if "frequencies" in name:
                        rest_params += [param]
                        continue
                    if "bias" in name:
                        rest_params += [param]
                        continue
                    adamW_params += [param]

            # AdamW optimizer
            AdamW = torch.optim.AdamW(
                adamW_params,
                lr=learning_rate,
                betas=(0.9, 0.999),
                eps=1e-07,
                weight_decay=weight_decay,
                amsgrad=True,
            )
            lr_schedule_AdamW = LinearWarmupExponentialDecay(
                AdamW, warmup_steps, decay_steps, decay_rate, staircase
            )

            # Adam: Optimzer for embeddings, frequencies and biases
            Adam = torch.optim.Adam(
                rest_params,
                lr=learning_rate,
                betas=(0.9, 0.999),
                eps=1e-07,
                amsgrad=True,
            )
            lr_schedule_Adam = LinearWarmupExponentialDecay(
                Adam, warmup_steps, decay_steps, decay_rate, staircase
            )

            # Wrap multiple optimizers to ease optimizer calls later on
            self.schedulers = MultiWrapper(
                lr_schedule_AdamW, lr_schedule_Adam
            )
            self.optimizers = MultiWrapper(AdamW, Adam)


        else:
            # Adam: Optimzer for all parameters
            Adam = torch.optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                betas=(0.9, 0.999),
                eps=1e-07,
                amsgrad=True,
            )
            lr_schedule_Adam = LinearWarmupExponentialDecay(
                Adam, warmup_steps, decay_steps, decay_rate, staircase
            )

            # Also wrap single optimizer for unified interface later
            self.schedulers = MultiWrapper(lr_schedule_Adam)
            self.optimizers = MultiWrapper(Adam)

        # Learning rate decay on plateau
        self.plateau_callback = ReduceLROnPlateau(
            optimizer=self.optimizers,
            scheduler=self.schedulers,
            factor=decay_factor,
            patience=decay_patience,
            cooldown=decay_cooldown,
            verbose=True,
        )

        if self.agc:
            # adaptive gradient clipping should not modify the last layer (see paper)
            self.params_except_last = []
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    if "out_energy" in name:
                        self.params_except_last += [param]
                    if "out_forces" in name:
                        self.params_except_last += [param]

        self.exp_decay = ExponentialMovingAverage(
            [p for p in self.model.parameters() if p.requires_grad], self.ema_decay
        )

    def save_variable_backups(self):
        self.exp_decay.store()

    def load_averaged_variables(self):
        self.exp_decay.copy_to()

    def restore_variable_backups(self):
        self.exp_decay.restore()

    def decay_maybe(self, val_loss):
        self.plateau_callback.step(val_loss)

    @staticmethod
    def _unitwise_norm(x, norm_type=2.0):
        if x.ndim <= 1:
            return x.norm(norm_type)
        else:
            # works for nn.ConvNd and nn,Linear where output dim is first in the kernel/weight tensor
            # might need special cases for other weights (possibly MHA) where this may not be true
            return x.norm(norm_type, dim=tuple(range(1, x.ndim)), keepdim=True)

    @staticmethod
    def _adaptive_gradient_clipping(parameters, clip_factor=0.05, eps=1e-3, norm_type=2.0):
        """
        https://github.com/rwightman/pytorch-image-models/blob/master/timm/utils/agc.py

        Adapted from High-Performance Large-Scale Image Recognition Without Normalization:
        https://github.com/deepmind/deepmind-research/blob/master/nfnets/optim.py"""
        with torch.no_grad():
            if isinstance(parameters, torch.Tensor):
                parameters = [parameters]
            for p in parameters:
                if p.grad is None:
                    continue
                p_data = p
                g_data = p.grad
                max_norm = (
                    Trainer._unitwise_norm(p_data, norm_type=norm_type)
                    .clamp_(min=eps)
                    .mul_(clip_factor)
                )
                grad_norm = Trainer._unitwise_norm(g_data, norm_type=norm_type)
                clipped_grad = g_data * (max_norm / grad_norm.clamp(min=1e-6))
                new_grads = torch.where(grad_norm < max_norm, g_data, clipped_grad)
                p.grad.copy_(new_grads)

    def scale_shared_grads(self):
        """Divide the gradients of the layers that are shared across multiple blocks
        by the number the weights are shared for
        """
        with torch.no_grad():

            def scale_grad(param, scale_factor):
                if param.grad is None:
                    return
                g_data = param.grad
                new_grads = g_data / scale_factor
                param.grad.copy_(new_grads)

            shared_int_layers = [
                self.model.mlp_rbf3,
                self.model.mlp_cbf3,
                self.model.mlp_rbf_h,
            ]
            if not self.model.triplets_only:
                shared_int_layers += [
                    self.model.mlp_rbf4,
                    self.model.mlp_cbf4,
                    self.model.mlp_sbf4,
                ]

            for layer in shared_int_layers:
                scale_grad(layer.weight, self.model.num_blocks)
            # output block is shared for +1 blocks
            scale_grad(self.model.mlp_rbf_out.weight, self.model.num_blocks + 1)

    def get_mae(self, targets, pred):
        """
        Mean Absolute Error
        """
        return torch.nn.functional.l1_loss(pred, targets, reduction="mean")

    def get_rmse(self, targets, pred):
        """
        Root Mean Squared Error
        """
        return torch.sqrt(torch.nn.functional.mse_loss(pred, targets, reduction="mean"))

    def get_nll(self, targets, mean_pred, var_pred):
        return torch.nn.functional.gaussian_nll_loss(
            mean_pred, targets, var_pred, reduction="mean"
        )

    def predict(self, inputs):

        energy, forces = self.model(inputs)

        if self.mve:
            mean_energy = energy[:, :1]
            var_energy = torch.nn.functional.softplus(energy[:, 1:])
            mean_forces = forces[:, 0, :]
            var_forces = torch.nn.functional.softplus(forces[:, 1, :])
            return mean_energy, var_energy, mean_forces, var_forces
        else:
            if len(forces.shape) == 3:
                forces = forces[:, 0]
            return energy, None, forces, None

    @staticmethod
    def dict2device(data, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for key in data:
            data[key] = data[key].to(device)
        return data

    def predict_on_batch(self, dataset_iter):
        inputs, _ = next(dataset_iter)
        inputs = self.dict2device(inputs)
        return self.predict(inputs)

    def train_on_batch(self, dataset_iter, metrics):
        self.model.train()
        inputs, targets = next(dataset_iter)
        # push to GPU if available
        inputs, targets = self.dict2device(inputs), self.dict2device(targets)

        mean_energy, var_energy, mean_forces, var_forces = self.predict(inputs)

        if self.mve:
            energy_nll = self.get_nll(targets["E"], mean_energy, var_energy)
            force_nll = self.get_nll(targets["F"], mean_forces, var_forces)
            loss = energy_nll * (1 - self.rho_force) + self.rho_force * force_nll
        else:
            energy_mae = self.get_mae(targets["E"], mean_energy)
            if self.loss == "mae":
                force_metric = self.get_mae(targets["F"], mean_forces)
            else:
                force_metric = self.get_rmse(targets["F"], mean_forces)
            loss = energy_mae * (1 - self.rho_force) + self.rho_force * force_metric

        self.optimizers.zero_grad()
        loss.backward()
        self.scale_shared_grads()

        if self.agc:
            self._adaptive_gradient_clipping(
                self.params_except_last, clip_factor=self.grad_clip_max
            )
        else:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.grad_clip_max
            )

        if self.mixed_precision:
            for opt in self.optimizers.wrapped:
                self.scaler.step(opt)
            self.scaler.update()
        else:
            self.optimizers.step()

        self.schedulers.step()
        self.exp_decay.update()

        # no gradients needed anymore
        loss = loss.detach()
        with torch.no_grad():
            if self.mve:
                energy_mae = self.get_mae(targets["E"], mean_energy)
                force_mae = self.get_mae(targets["F"], mean_forces)
                force_rmse = self.get_rmse(targets["F"], mean_forces)

            else:
                if self.loss == "mae":
                    force_mae = force_metric
                    force_rmse = self.get_rmse(targets["F"], mean_forces)
                else:
                    force_mae = self.get_mae(targets["F"], mean_forces)
                    force_rmse = force_metric

            if self.mve:
                # update molecule metrics
                metrics.update_state(
                    nsamples=mean_energy.shape[0],
                    loss=loss,
                    energy_mae=energy_mae,
                    energy_nll=energy_nll,
                    energy_var=var_energy,
                )
                # update atom metrics
                metrics.update_state(
                    nsamples=mean_forces.shape[0],
                    force_mae=force_mae,
                    force_rmse=force_rmse,
                    force_nll=force_nll,
                    force_var=var_forces,
                )
            else:
                # update molecule metrics
                metrics.update_state(
                    nsamples=mean_energy.shape[0],
                    loss=loss,
                    energy_mae=energy_mae,
                )
                # update atom metrics
                metrics.update_state(
                    nsamples=mean_forces.shape[0],
                    force_mae=force_mae,
                    force_rmse=force_rmse,
                )

        return loss

    def test_on_batch(self, dataset_iter, metrics):
        self.model.eval()
        inputs, targets = next(dataset_iter)
        # push to GPU if available
        inputs, targets = self.dict2device(inputs), self.dict2device(targets)

        if self.model.direct_forces:
            # do not need any gradients -> reduce memory consumption
            with torch.no_grad():
                mean_energy, var_energy, mean_forces, var_forces = self.predict(inputs)
        else:
            # need gradient for forces
            mean_energy, var_energy, mean_forces, var_forces = self.predict(inputs)

        with torch.no_grad():
            energy_mae = self.get_mae(targets["E"], mean_energy)
            force_mae = self.get_mae(targets["F"], mean_forces)
            force_rmse = self.get_rmse(targets["F"], mean_forces)

            if self.mve:
                energy_nll = self.get_nll(targets["E"], mean_energy, var_energy)
                loss = energy_nll * (1 - self.rho_force) + self.rho_force * force_mae
                force_nll = self.get_nll(targets["F"], mean_forces, var_forces)
                loss = energy_nll * (1 - self.rho_force) + self.rho_force * force_nll

                # update molecule metrics
                metrics.update_state(
                    nsamples=mean_energy.shape[0],
                    loss=loss,
                    energy_mae=energy_mae,
                    energy_nll=energy_nll,
                    energy_var=var_energy,
                )
                # update atom metrics
                metrics.update_state(
                    nsamples=mean_forces.shape[0],
                    force_mae=force_mae,
                    force_rmse=force_rmse,
                    force_nll=force_nll,
                    force_var=var_forces,
                )

            else:
                if self.loss == "mae":
                    force_metric = self.get_mae(targets["F"], mean_forces)
                else:
                    force_metric = self.get_rmse(targets["F"], mean_forces)

                loss = energy_mae * (1 - self.rho_force) + self.rho_force * force_metric

                # update molecule metrics
                metrics.update_state(
                    nsamples=mean_energy.shape[0],
                    loss=loss,
                    energy_mae=energy_mae,
                )
                # update atom metrics
                metrics.update_state(
                    nsamples=mean_forces.shape[0],
                    force_mae=force_mae,
                    force_rmse=force_rmse,
                )

        return loss

    def eval_on_batch(self, dataset_iter):
        self.model.eval()
        with torch.no_grad():
            inputs, targets = next(dataset_iter)
            # push to GPU if available
            inputs, targets = self.dict2device(inputs), self.dict2device(targets)
            energy, _, forces, _ = self.predict(inputs)
        return (energy, forces), targets

    def state_dict(self):
        """Returns the state of the trainer and all subinstancces except the model."""
        state_dict = {
            key: value
            for key, value in self.__dict__.items()
            if key
            not in [
                "model",
                "schedulers",
                "optimizers",
                "plateau_callback",
                "exp_decay",
            ]
        }
        for attr in ["schedulers", "optimizers", "plateau_callback", "exp_decay"]:
            state_dict.update({attr: getattr(self, attr).state_dict()})
        return state_dict

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        trainer_dict = {
            key: value
            for key, value in self.state_dict.items()
            if key
            not in [
                "model",
                "schedulers",
                "optimizers",
                "plateau_callback",
                "exp_decay",
            ]
        }
        self.__dict__.update(trainer_dict)
        for attr in ["schedulers", "optimizers", "plateau_callback", "exp_decay"]:
            getattr(self, attr).load_state_dict(state_dict[attr])


class ReduceLROnPlateau:
    """Reduce learning rate (and weight decay) when a metric has stopped improving.
    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates. This scheduler reads a metrics
    quantity and if no improvement is seen for a 'patience' number
    of steps, the learning rate (and weight decay) is reduced.

    Parameters
    ----------
        optimizer: Optimizer, list:
            Wrapped optimizer.
        scheduler: LRSchedule, list
            Learning rate schedule of the optimizer.
            Asserts that the second schedule belongs to second optimizer and so on.
        mode: str
            One of `min`, `max`. In `min` mode, lr will
            be reduced when the quantity monitored has stopped
            decreasing; in `max` mode it will be reduced when the
            quantity monitored has stopped increasing. Default: 'min'.
        factor: float
            Factor by which the learning rate will be
            reduced. new_lr = lr * factor. Default: 0.1.
        patience: int
            Number of steps with no improvement after
            which learning rate will be reduced. For example, if
            `patience = 2`, then we will ignore the first 2 steps
            with no improvement, and will only decrease the LR after the
            3rd step if the loss still hasn't improved then.
            Default: 10.
        threshold: float
            Threshold for measuring the new optimum,
            to only focus on significant changes. Default: 1e-4.
        max_reduce: int
            Number of maximum decays on plateaus. Default: 10.
        threshold_mode: str
            One of `rel`, `abs`. In `rel` mode,
            dynamic_threshold = best * ( 1 + threshold ) in 'max'
            mode or best * ( 1 - threshold ) in `min` mode.
            In `abs` mode, dynamic_threshold = best + threshold in
            `max` mode or best - threshold in `min` mode. Default: 'rel'.
        cooldown: int
            Number of steps to wait before resuming
            normal operation after lr has been reduced. Default: 0.
        eps: float
            Minimal decay applied to lr. If the difference
            between new and old lr is smaller than eps, the update is
            ignored. Default: 1e-8.
        verbose: bool
            If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    """

    def __init__(
        self,
        optimizer,
        scheduler,
        factor=0.1,
        patience=10,
        threshold=1e-4,
        max_reduce=10,
        cooldown=0,
        threshold_mode="rel",
        min_lr=0,
        eps=1e-8,
        mode="min",
        verbose=False,
    ):

        if factor >= 1.0:
            raise ValueError(f"Factor should be < 1.0 but is {factor}.")
        self.factor = factor
        self.optimizer = optimizer
        self.scheduler = scheduler

        if isinstance(optimizer, MultiWrapper):
            self.optimizer = optimizer.wrapped
        if isinstance(scheduler, MultiWrapper):
            self.scheduler = scheduler.wrapped

        if not isinstance(self.optimizer, (list,tuple)):
            self.optimizer = [self.optimizer]
        if not isinstance(self.scheduler, (list,tuple)):
            self.scheduler = [self.scheduler]

        assert len(self.optimizer) == len(self.scheduler)

        for opt in self.optimizer:
            # Attach optimizer
            if not isinstance(opt, torch.optim.Optimizer):
                raise TypeError(f"{type(opt).__name__} is not an Optimizer but is of type {type(opt)}")

        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_steps = None
        self.mode_worse = None  # the worse value for the chosen mode
        self.eps = eps
        self.last_step = 0
        self._init_is_better(
            mode=mode, threshold=threshold, threshold_mode=threshold_mode
        )
        self._reset()
        self._reduce_counter = 0

    def _reset(self):
        """Resets num_bad_steps counter and cooldown counter."""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_steps = 0

    def step(self, metrics):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        step = self.last_step + 1
        self.last_step = step

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_steps = 0
        else:
            self.num_bad_steps += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_steps = 0  # ignore any bad steps in cooldown

        if self.num_bad_steps > self.patience:
            self._reduce(step)
            self.cooldown_counter = self.cooldown
            self.num_bad_steps = 0

    def _reduce(self, step):
        self._reduce_counter += 1

        for optimzer, schedule in zip(self.optimizer, self.scheduler):
            if hasattr(schedule, "base_lrs"):
                schedule.base_lrs = [lr * self.factor for lr in schedule.base_lrs]
            else:
                raise ValueError(
                    "Schedule does not have attribute 'base_lrs' for the learning rate."
                )
        if self.verbose:
            logging.info(f"Step {step}: reducing on plateu by {self.factor}.")

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def is_better(self, a, best):
        if self.mode == "min" and self.threshold_mode == "rel":
            rel_epsilon = 1.0 - self.threshold
            return a < best * rel_epsilon

        elif self.mode == "min" and self.threshold_mode == "abs":
            return a < best - self.threshold

        elif self.mode == "max" and self.threshold_mode == "rel":
            rel_epsilon = self.threshold + 1.0
            return a > best * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + self.threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {"min", "max"}:
            raise ValueError("mode " + mode + " is unknown!")
        if threshold_mode not in {"rel", "abs"}:
            raise ValueError("threshold mode " + threshold_mode + " is unknown!")

        if mode == "min":
            self.mode_worse = np.inf
        else:  # mode == 'max':
            self.mode_worse = -np.inf

        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode

    def state_dict(self):
        return {
            key: value
            for key, value in self.__dict__.items()
            if key not in ["optimizer", "scheduler"]
        }

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
        self._init_is_better(
            mode=self.mode, threshold=self.threshold, threshold_mode=self.threshold_mode
        )


class MultiWrapper:
    def __init__(self, *ops):
        self.wrapped = ops

    def __getitem__(self, idx):
        return self.wrapped[idx]

    def zero_grad(self):
        for op in self.wrapped:
            op.zero_grad()

    def step(self):
        for op in self.wrapped:
            op.step()

    def state_dict(self):
        """Returns the overall state dict of the wrapped instances."""
        return {i: opt.state_dict() for i, opt in enumerate(self.wrapped)}

    def load_state_dict(self, state_dict):
        """Load the state_dict for each wrapped instance.
        Assumes the order is the same as when the state_dict was loaded
        """
        for i, opt in enumerate(self.wrapped):
            opt.load_state_dict(state_dict[i])
