from .raft_trainer import RaftTrainer
import torch.nn.functional as F
import torch

def compute_mrt_loss(rewards, logits, labels, n_cand, mrt_alpha):
    """
    Args:
        rewards (torch.Tensor): [batch_size * n_cand]
        logits (torch.Tensor): [batch_size * n_cand, max_length, vocab_size]
        labels (torch.Tensor): [batch_size * n_cand, max_length]
        n_cand (int): # candidaites per sample
        mrt_alpha (float): hyperparameter 
    """
    assert (len(rewards) % n_cand) == 0
    batch_size = int(len(rewards) / n_cand)
    vocab_size = logits.shape[-1]
    max_length = labels.shape[-1]

    loss_per_token = F.cross_entropy(
        logits.view(-1, vocab_size),
        labels.view(-1),
        reduction='none'
    ).view(batch_size * n_cand, max_length) # [batch_size * n_cand, vocab_size]

    loss_per_sent = loss_per_token.sum(dim=1) # [batch_size * n_cand]
    lprob_per_sent = -loss_per_sent * mrt_alpha # [batch_size * n_cand]
    norm_lprob_per_sent = F.softmax(lprob_per_sent.view(batch_size, n_cand), dim=-1).view(-1) # [batch_size * n_cand]
    loss = torch.dot(norm_lprob_per_sent, 1 - rewards) / batch_size
    return loss

class MRTTrainer(RaftTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        reward = inputs.pop("reward") # [batch_size * n_cand]
        n_cand = int(inputs.pop("n_cand")[0])
        outputs = model(**inputs) # outputs.logits: [batch_size * n_cand, max_length]

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        loss = compute_mrt_loss(reward, outputs["logits"], inputs["labels"], n_cand, self.args.mrt_alpha)

        return (loss, outputs) if return_outputs else loss