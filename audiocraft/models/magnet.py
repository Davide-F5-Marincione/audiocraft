# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Main model for using MAGNeT. This will combine all the required components
and provide easy access to the generation API.
"""
import typing as tp
import torch

from .genmodel import BaseGenModel
from .lm import LMModel
from .loaders import load_compression_model, load_lm_model_magnet


class MAGNeT(BaseGenModel):
    """MAGNeT main model with convenient generation API.
    Args:
       See MusicGen class.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # MAGNeT operates over a fixed sequence length defined in it's config.
        self.duration = self.lm.cfg.dataset.segment_duration
        self.set_generation_params()

    @staticmethod
    def get_pretrained(name: str = 'facebook/magnet-small-10secs', device=None):
        """Return pretrained model, we provide six models:
        - facebook/magnet-small-10secs (300M), text to music, 10-second audio samples.
          # see: https://huggingface.co/facebook/magnet-small-10secs
        - facebook/magnet-medium-10secs (1.5B), text to music, 10-second audio samples.
          # see: https://huggingface.co/facebook/magnet-medium-10secs
        - facebook/magnet-small-30secs (300M), text to music, 30-second audio samples.
          # see: https://huggingface.co/facebook/magnet-small-30secs
        - facebook/magnet-medium-30secs (1.5B), text to music, 30-second audio samples.
          # see: https://huggingface.co/facebook/magnet-medium-30secs
        - facebook/audio-magnet-small (300M), text to sound-effect (10-second samples).
          # see: https://huggingface.co/facebook/audio-magnet-small
        - facebook/audio-magnet-medium (1.5B), text to sound-effect (10-second samples).
          # see: https://huggingface.co/facebook/audio-magnet-medium
        """
        if device is None:
            if torch.cuda.device_count():
                device = 'cuda'
            else:
                device = 'cpu'

        compression_model = load_compression_model(name, device=device)
        lm = load_lm_model_magnet(name, compression_model_frame_rate=int(compression_model.frame_rate), device=device)

        if 'self_wav' in lm.condition_provider.conditioners:
            lm.condition_provider.conditioners['self_wav'].match_len_on_eval = True

        kwargs = {'name': name, 'compression_model': compression_model, 'lm': lm}
        return MAGNeT(**kwargs)

    def set_generation_params(self, use_sampling: bool = True, top_k: int = 0,
                              top_p: float = 0.9, temperature: float = 3.0,
                              max_cfg_coef: float = 10.0, min_cfg_coef: float = 1.0,
                              decoding_steps: tp.List[int] = [20, 10, 10, 10],
                              span_arrangement: str = 'nonoverlap',
                              rescorer: LMModel = None,
                              rescore_weights: torch.Tensor | float = 0.7,
                              rescorer_temp: torch.Tensor | float = 1.0,
                              loop_size_perc: float = 0.0,
                              k_loops: int = 1):
        """Set the generation parameters for MAGNeT.

        Args:
            use_sampling (bool, optional): Use sampling if True, else do argmax decoding. Defaults to True.
            top_k (int, optional): top_k used for sampling. Defaults to 0.
            top_p (float, optional): top_p used for sampling, when set to 0 top_k is used. Defaults to 0.9.
            temperature (float, optional): Initial softmax temperature parameter. Defaults to 3.0.
            max_cfg_coef (float, optional): Coefficient used for classifier free guidance. Defaults to 10.0.
            min_cfg_coef (float, optional): End coefficient of classifier free guidance annealing. Defaults to 1.0.
            decoding_steps (list of n_q ints, optional): The number of iterative decoding steps,
                                                         for each of the n_q RVQ codebooks.
            span_arrangement (str, optional): Use either non-overlapping spans ('nonoverlap')
                                              or overlapping spans ('stride1') in the masking scheme.
        """
        if isinstance(rescore_weights, int):
          rescore_weights = float(rescore_weights)

        self.k_loops = k_loops
        self.generation_params = {
            'use_sampling': use_sampling,
            'temp': temperature,
            'top_k': top_k,
            'top_p': top_p,
            'max_cfg_coef': max_cfg_coef,
            'min_cfg_coef': min_cfg_coef,
            'decoding_steps': [int(s) for s in decoding_steps],
            'span_arrangement': span_arrangement,
            'rescorer': rescorer,
            'rescore_weights': rescore_weights,
            'rescorer_temp': rescorer_temp,
            'loop_size_perc': loop_size_perc
        }

    def generate_audio(self, gen_tokens: torch.Tensor) -> torch.Tensor:
        """Generate Audio from tokens."""
        assert gen_tokens.dim() == 3
        with torch.no_grad():
            # From DAVIDE
            left_pad = 0
            right_pad = 0
            if self.generation_params['loop_size_perc'] > 0:
                valid_amount = int(gen_tokens.shape[-1] * self.generation_params['loop_size_perc'])
                empty_amount = gen_tokens.shape[-1]  - valid_amount
                left_pad = min(valid_amount, empty_amount // 2)
                right_pad = empty_amount - left_pad
            
            gen_audio = self.compression_model.decode(gen_tokens, None)

            # From DAVIDE
            if right_pad > 0:
                gen_audio = gen_audio[..., 640 * left_pad: -640*right_pad]

            gen_audio = torch.cat([gen_audio] * self.k_loops, -1)
        return gen_audio