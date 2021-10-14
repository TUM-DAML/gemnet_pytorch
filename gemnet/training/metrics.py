import numpy as np
import os
import logging


class BestMetrics:
    """Class for saving the metrics.

    Parameters
    ----------
        path: str
            Directory where to save the results in.
        metic: Metrics
            instance to save the best state of.
        assert_exist: bool
            If True raise UserWarning if the metrics should be restored but None are found.
            If False log Warning and frehsly initilaize the metrics.
    """

    def __init__(self, path, metrics, assert_exist=True):
        self.path = os.path.join(path, "best_metrics.npz")
        self.metrics = metrics
        self.assert_exist = assert_exist
        self.state = {}

    def inititalize(self):
        self.state = {f"{k}_{self.metrics.tag}": np.inf for k in self.metrics.keys}
        self.state["step"] = 0
        np.savez(self.path, **self.state)

    def restore(self):
        if not os.path.isfile(self.path):
            string = f"Best metrics can not be restored as the file does not exist in the given path: {self.path}"
            if self.assert_exist:
                raise UserWarning(string)

            string += "\n Will initialize the best metrics."
            logging.warning(string)
            self.inititalize()
        else:
            loss_file = np.load(self.path)
            self.state = {k: v.item() for k, v in loss_file.items()}

    def items(self):
        return self.state.items()

    def update(self, step, metrics):
        self.state["step"] = step
        self.state.update(metrics.result())
        np.savez(self.path, **self.state)

    def write(self, summary_writer, step):
        for key, val in self.state.items():
            if key != "step":
                summary_writer.add_scalar(key + "_best", val, step)

    @property
    def loss(self):
        return self.state["loss_val"]

    @property
    def step(self):
        return self.state["step"]


class MeanMetric:
    def __init__(self):
        self.reset_states()

    def update_state(self, values, sample_weight):
        self.values += sample_weight * values
        self.sample_weights += sample_weight

    def result(self):
        return self.values / self.sample_weights

    def reset_states(self):
        self.sample_weights = 0
        self.values = 0


class Metrics:
    """Class for saving the metrics.

    Parameters
    ----------
        tag: str
            Tag to add to the metric (e.g 'train' or 'val').
        keys: list
            Name of the different metrics to watch (e.g. 'loss', 'mae' etc)
        ex: sacred.Eperiment
            Sacred experiment that keeps track of the metrics.
    """

    def __init__(self, tag, keys, ex=None):
        self.tag = tag
        self.keys = keys
        self.ex = ex

        assert "loss" in self.keys
        self.mean_metrics = {}
        for key in self.keys:
            self.mean_metrics[key] = MeanMetric()

    def update_state(self, nsamples, **updates):
        """Update the metrics.

        Parameters
        ----------
            nsamples: int
                Number of samples for which the updates where calculated on.
            updates: dict
                Contains metric updates.
        """
        assert set(updates.keys()).issubset(set(self.keys))
        for key in updates:
            self.mean_metrics[key].update_state(
                updates[key].cpu(), sample_weight=nsamples
            )

    def write(self, summary_writer, step):
        """Write metrics to summary_writer (and the Sacred experiment)."""
        for key, val in self.result().items():
            summary_writer.add_scalar(key, val, global_step=step)
            if self.ex is not None:
                if key not in self.ex.current_run.info:
                    self.ex.current_run.info[key] = []
                self.ex.current_run.info[key].append(val)

        if self.ex is not None:
            if f"step_{self.tag}" not in self.ex.current_run.info:
                self.ex.current_run.info[f"step_{self.tag}"] = []
            self.ex.current_run.info[f"step_{self.tag}"].append(step)

    def reset_states(self):
        for key in self.keys:
            self.mean_metrics[key].reset_states()

    def result(self, append_tag=True):
        """
        Parameters
        ----------
            append_tag: bool
                If True append the tag to the key of the returned dict

        Returns
        -------
            result_dict: dict
                Contains the numpy values of the metrics.
        """
        result_dict = {}
        for key in self.keys:
            result_key = f"{key}_{self.tag}" if append_tag else key
            result_dict[result_key] = self.mean_metrics[key].result().numpy().item()
        return result_dict

    @property
    def loss(self):
        return self.mean_metrics["loss"].result().numpy().item()
