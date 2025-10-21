from pytorch_lightning.callbacks import Callback
import torch


class TerminateOnNaN(Callback):
    """
    Callback to terminate training if the loss becomes NaN or Inf.
    """
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = outputs.get('loss')
        
        if loss is None:
            loss = trainer.callback_metrics.get("train_loss_step")

        if loss is not None and (torch.isnan(loss) or torch.isinf(loss)):
            print("\nLoss is NaN or Inf. Terminating training.")
            trainer.should_stop = True