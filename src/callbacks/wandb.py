import wandb
from catalyst import dl


class WandbCallback(dl.Callback):
    def __init__(self):
        super().__init__(order=dl.CallbackOrder.Logging)

    def on_loader_end(self, runner):
        loader = runner.loader_key
        metrics = runner.loader_metrics
        to_log = {}
        for metric_key, metric in metrics.items():
            to_log[f"{loader}_{metric_key}"] = metric
        wandb.log(to_log, commit=False)

    def on_epoch_end(self, runner):
        wandb.log({}, commit=True)

    def on_stage_end(self, runner):
        artifact = wandb.Artifact('best-model', type='model')
        logdir = str(runner.logdir)
        model_path = logdir + "/checkpoints/best.pth"
        artifact.add_file(model_path, name="best_model")