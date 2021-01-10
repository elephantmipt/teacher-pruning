import torch
from typing import Mapping, Any, Callable
from collections import OrderedDict

from catalyst.dl import Runner
from catalyst.experiments.experiment import Experiment
from catalyst.typing import RunnerModel, Device
from catalyst.utils import get_nn_from_ddp_module, set_requires_grad


class KDRunner(Runner):
    def __init__(
            self,
            model: RunnerModel = None,
            device: Device = None,
            experiment_fn: Callable = Experiment,
            output_hiddens: bool = False,
    ):
        """

        Args:
            model: Torch model object
            device: Torch device
            experiment_fn: callable function,
                which defines default experiment type to use
                during ``.train`` and ``.infer`` methods.
            output_hiddens: flag to output hidden states during training
        """
        super().__init__(model=model, device=device, experiment_fn=experiment_fn)
        self.output_hiddens = output_hiddens

    def _handle_batch(self, batch: Mapping[str, Any]) -> None:
        self.output = OrderedDict()
        need_hiddens = self.is_train_loader and self.output_hiddens
        student = get_nn_from_ddp_module(self.model["student"])
        teacher = get_nn_from_ddp_module(self.model["teacher"])
        teacher.eval()
        set_requires_grad(teacher, False)
        s_outputs = student(
            batch["features"], output_hiddens=need_hiddens
        )
        t_outputs = teacher(
            batch["features"], output_hiddens=need_hiddens
        )
        if need_hiddens:
            self.output["logits"] = s_outputs[0]
            self.output["hiddens"] = s_outputs[1]
            self.output["teacher_logits"] = t_outputs[0]
            self.output["teacher_hiddens"] = t_outputs[1]
        else:
            self.output["logits"] = s_outputs
            self.output["teacher_logits"] = t_outputs
