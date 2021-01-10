from torch import Tensor
from torch import nn
from torch.nn import functional as F
from catalyst.core import Callback, CallbackOrder

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from catalyst.core import IRunner


class DiffLossCallback(Callback):
    def __init__(
            self,
            criterion: str = None,
            temperature: float = 1.,
            metric_key: str = "diff_loss"
    ):
        """
        KL Div loss on output callback.
        Args:
            criterion: criterion for loss on outputs.
            Can be kl, mse or cos.
            temperature: temperature for logits.
            metric_key: key for metric in batch_metrics dict.
        Raises:
            TypeError: if criterion is not correct.
        """
        super().__init__(CallbackOrder.Metric)
        if criterion is None:
            criterion = "kl"
        self.criterion = criterion
        if criterion == "kl":
            self.criterion_fn = nn.KLDivLoss()
            self.temperature = temperature
        elif criterion == "mse":
            self.criterion_fn = nn.MSELoss(reduction="sum")
        elif criterion == "cos":
            self.criterion_fn = nn.CosineEmbeddingLoss(reduction="mean")
        else:
            raise TypeError(f"Criterion should be string one of the kl, mse or cos")
        if not (self.temperature == 1. or self.criterion == "kl"):
            Warning("Temperature affects only if criterion is kl")
        self.metric_key = metric_key

    def _calculate_loss(self, student_output, teacher_output) -> Tensor:
        if self.criterion == "kl":
            loss = (
                    self.criterion_fn(
                        F.log_softmax(student_output / self.temperature, dim=-1),
                        F.softmax(teacher_output / self.temperature, dim=-1),
                    )
                    * self.temperature ** 2
            )
        elif self.criterion == "mse":
            loss = (
                    self.criterion_fn(student_output, teacher_output)
                    / student_output.size(0)  # Reproducing batchmean reduction
            )
        else:
            loss = self.criterion_fn(student_output, teacher_output)
        return loss


class DiffOutputCallback(DiffLossCallback):

    def __init__(
            self,
            criterion: str = None,
            temperature: float = 1.,
            metric_key: str = "diff_output_loss"
    ):
        """
        KL Div loss on output callback.
        Args:
            criterion: criterion for loss on outputs.
            Can be kl, mse or cos.
            temperature: temperature for logits.
            metric_key: key for metric in batch_metrics dict.
        Raises:
            TypeError: if criterion is not correct.
        """
        super().__init__(
            criterion=criterion,
            temperature=temperature,
            metric_key=metric_key,
        )

    def on_batch_end(self, runner: "IRunner") -> None:
        """
        On batch end action. Calculates difference between
        teacher's and student's logits.
        Args:
            runner: runner for experiment.
        """
        if runner.is_train_loader:
            teacher_logits = runner.output["teacher_logits"]
            student_logits = runner.output["logits"]

            runner.batch_metrics["output_diff_loss"] = \
                self._calculate_loss(
                    student_output=student_logits,
                    teacher_output=teacher_logits
                )


class DiffHiddenCallback(DiffLossCallback):

    def __init__(
            self,
            criterion: str = None,
            temperature: float = 1.,
            metric_key: str = "diff_hidden_loss"
    ):
        """
        KL Div loss on output callback.
        Args:
            criterion: criterion for loss on outputs.
            Can be kl, mse or cos.
            temperature: temperature for logits.
            metric_key: key for metric in batch_metrics dict.
        Raises:
            TypeError: if criterion is not correct.
        """
        if criterion is None:
            criterion = "mse"
        super().__init__(
            criterion=criterion,
            temperature=temperature,
            metric_key=metric_key
        )
        if self.criterion == "kl":
            Warning(
                "Probably hidden states is not logits for density function,"
                "so please don't use kl-divergence"
            )

    def on_batch_end(self, runner: "IRunner") -> None:
        """
        On batch end action. Calculates difference between
        teacher's and student's logits.
        Args:
            runner: runner for experiment.

        Raises:
            TypeError: if length of hiddens is not equal.
        """
        if runner.is_train_loader:
            teacher_hiddens: List[Tensor] = runner.output["teacher_hiddens"]
            student_hiddens: List[Tensor] = runner.output["hiddens"]
            if len(student_hiddens) != len(teacher_hiddens):
                raise TypeError(
                    "Student's and teacher's hiddens "
                    "should be the same length. "
                    "Got {} for student and {} for teacher".format(
                        len(student_hiddens), len(teacher_hiddens)
                    )
                )
            loss = 0
            for c_student_hidden, c_teacher_hidden in \
                    zip(student_hiddens, teacher_hiddens):
                loss += \
                    self._calculate_loss(c_student_hidden, c_teacher_hidden) \
                    / len(student_hiddens)
            runner.batch_metrics["output_diff_loss"] = loss


__all__ = ["DiffOutputCallback", "DiffHiddenCallback"]
