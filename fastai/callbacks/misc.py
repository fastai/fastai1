" Miscellaneous callbacks "

from ..basic_train import LearnerCallback
from fastai.utils.mod_display import progress_disabled_ctx

class StopAfterNBatches(LearnerCallback):
    "Stop training after n batches of the first epoch."
    def __init__(self, learn, n_batches:int=2):
        super().__init__(learn)
        self.n_batches = n_batches-1 # iteration starts from 0
        # XXX: enable later, see below
        # self.prog_ctx = progress_disabled_ctx(learn)

    def on_batch_end(self, iteration, **kwargs):
        if iteration == self.n_batches:
            return {'stop_epoch': True, 'stop_training': True, 'skip_validate': True}

    # XXX: enable these after part 2 is over and Sylvain fixed fastprogress to support that - currently enabling the context for this callback in lesson7-superres.ipynb breaks in the fastprogress domain during lr_find
    # enable clean early stopping without red interrupted progress bars
    #def on_train_begin(self, **kwargs): self.prog_ctx.disable()
    #def on_train_end(self, **kwargs):   self.prog_ctx.enable()
