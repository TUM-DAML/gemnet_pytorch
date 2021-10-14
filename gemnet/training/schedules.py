from torch.optim.lr_scheduler import LambdaLR

class LinearWarmupExponentialDecay(LambdaLR):
    """This schedule combines a linear warmup with an exponential decay.

    Parameters
    ----------
        optimizer: Optimizer
            Optimizer instance.
        decay_steps: float
            Number of steps until learning rate reaches learning_rate*decay_rate
        decay_rate: float
            Decay rate.
        warmup_steps: int
            Total number of warmup steps of the learning rate schedule.
        staircase: bool
            If True use staircase decay and not (continous) exponential decay.
        last_step: int
            Only needed when resuming training to resume learning rate schedule at this step.
    """

    def __init__(
        self,
        optimizer,
        warmup_steps,
        decay_steps,
        decay_rate,
        staircase=False,
        last_step=-1,
        verbose=False,
    ):
        assert decay_rate <= 1

        if warmup_steps == 0:
            warmup_steps = 1

        def lr_lambda(step):
            # step starts at 0
            warmup = min(1 / warmup_steps + 1 / warmup_steps * step, 1)
            exponent = step / decay_steps
            if staircase:
                exponent = int(exponent)
            decay = decay_rate ** exponent
            return warmup * decay

        super().__init__(optimizer, lr_lambda, last_epoch=last_step, verbose=verbose)
