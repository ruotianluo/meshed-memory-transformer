import torch
from torch import nn
import copy

from .mytrans.models.TransformerModel import TransformerModel
from m2transformer.models.captioning_model import CaptioningModel
import m2transformer.utils as utils

class Config(dict):
    def __getattr__(self, k):
        if k in self:
            return self[k]
        else:
            raise AttributeError
    def __setattr__(self, k, v):
        self[k] = v


class OriginalTransformer(CaptioningModel):
    def __init__(self, bos_idx, eos_idx, pad_idx, vocab):
        super(OriginalTransformer, self).__init__()
        self.bos_idx = bos_idx

        config = Config()
        config.vocab_size = len(vocab) - 1 # This is mine assuming no bos eos and pad
        config.vocab = vocab

        config.input_encoding_size = 1
        #self.rnn_type = opt.rnn_type
        config.rnn_size = 1
        config.num_layers = 1
        config.drop_prob_lm = 0.5
        config.seq_length = 20
        config.fc_feat_size = 2048
        config.att_feat_size = 2048
        config.att_hid_size = 1

        config.N_enc = 6
        config.N_dec = 6
        config.d_model = 512
        config.d_ff = 2048
        config.h = 8
        config.dropout = 0.1

        config.bos_idx = bos_idx
        config.eos_idx = eos_idx
        config.pad_idx = pad_idx

        self.d_model = config.d_model
        self.model = TransformerModel(config)


    def forward(self, images, seq, *args):
        image_mask = images.sum(-1) != 0
        out = self.model(images.new_zeros(images.shape[0], 0),
                          images, seq, image_mask, mode='forward')
        return out

    def sample_rl(self, visual: utils.TensorOrSequence, max_len: int, **kwargs) -> utils.Tuple[torch.Tensor, torch.Tensor]:
        b_s = utils.get_batch_size(visual)
        outputs = []
        log_probs = []

        with self.statefulness(b_s):
            out = None
            for t in range(max_len):
                out = self.step(t, out, visual, None, mode='feedback', **kwargs)
                distr = distributions.Categorical(logits=out[:, 0])
                out = distr.sample().unsqueeze(1)
                outputs.append(out)
                log_probs.append(distr.log_prob(out).unsqueeze(1))

        return torch.cat(outputs, 1), torch.cat(log_probs, 1)

    def sample_n_rl(self, visual: utils.TensorOrSequence, max_len: int, eos_idx: int, sample_size=1, **kwargs):
        self.model.seq_length = max_len

        image_mask = visual.sum(-1) != 0

        gen_result, sample_logprobs = self.model(
            visual.new_zeros(visual.shape[0], 0),
            visual, image_mask,
            opt={'sample_method':'sample',
                 'output_logsoftmax': True,
                 'sample_n': sample_size},
            mode='sample')

        logprobs = sample_logprobs.gather(2, gen_result.unsqueeze(2)).squeeze(2)

        seq = gen_result.reshape(visual.shape[0], sample_size, -1)
        logprobs = logprobs.reshape(visual.shape[0], sample_size, -1)
        sample_logprobs = sample_logprobs.reshape(visual.shape[0], sample_size, *sample_logprobs.shape[-2:])

        # seq = seq[:,:out_size]
        # logprobs = logprobs[:,:out_size]
        # sample_logprobs = sample_logprobs[:,:out_size]
        # if out_size == 1:
        #     seq = seq.squeeze(1)
        #     logprobs = logprobs.squeeze(1)
        #     sample_logprobs = sample_logprobs.squeeze(1)

        # if return_probs:
        #     return seq, logprobs, sample_logprobs
        # else:
        return seq, logprobs

    def beam_search(self, visual: utils.TensorOrSequence, max_len: int, eos_idx: int, beam_size: int, out_size=1,
                    return_probs=False, **kwargs):
        self.model.seq_length = max_len

        image_mask = visual.sum(-1) != 0

        gen_result, sample_logprobs = self.model(
            visual.new_zeros(visual.shape[0], 0),
            visual, image_mask,
            opt={'sample_method':'greedy',
                 'beam_size':beam_size,
                 'output_logsoftmax': True,
                 'sample_n': beam_size},
            mode='sample')

        logprobs = sample_logprobs.gather(2, gen_result.unsqueeze(2)).squeeze(2)

        seq = gen_result.reshape(visual.shape[0], beam_size, -1)
        logprobs = logprobs.reshape(visual.shape[0], beam_size, -1)
        sample_logprobs = sample_logprobs.reshape(visual.shape[0], beam_size, *sample_logprobs.shape[-2:])

        seq = seq[:,:out_size]
        logprobs = logprobs[:,:out_size]
        sample_logprobs = sample_logprobs[:,:out_size]
        if out_size == 1:
            seq = seq.squeeze(1)
            logprobs = logprobs.squeeze(1)
            sample_logprobs = sample_logprobs.squeeze(1)

        if return_probs:
            return seq, logprobs, sample_logprobs
        else:
            return seq, logprobs
