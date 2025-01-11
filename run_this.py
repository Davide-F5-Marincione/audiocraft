import torch
import torchaudio
import numpy as np
import sys
from audiocraft.models import MusicGen, MAGNeT
from audiocraft.data.audio import audio_write

seed = 42
loop_setup = True
sec30 = False

torch.manual_seed(seed)
np.random.seed(seed)

assert torch.cuda.is_available(), "Man, you NEED a GPU for this."
device = "cuda"
rescorer = MusicGen.get_pretrained('facebook/musicgen-medium', device=device).lm
if sec30:
    model = MAGNeT.get_pretrained('facebook/magnet-medium-30secs', device=device)
else:
    model = MAGNeT.get_pretrained('facebook/magnet-medium-10secs', device=device)


first_dec = 20 * (3 if sec30 else 1)

if loop_setup:
    model.set_generation_params(
        span_arrangement = 'stride1',
        use_sampling = True,
        top_k=0,
        top_p = .9,
        temperature = 3.0,
        max_cfg_coef = 10.0,
        min_cfg_coef = 1.0,
        decoding_steps = [first_dec, 10, 10, 10],
        rescorer = rescorer,
        rescore_weights = 0.7,
        loop_trick_rotations = 4,
        loop_trick_aggregation = 'prod_probs',
        loop_pad=20,
        seam_enforce=20,
        random_roll=False
    )
else:
    model.set_generation_params(
        span_arrangement = 'stride1',
        use_sampling = True,
        top_k=0,
        top_p = .9,
        temperature = 3.0,
        max_cfg_coef = 10.0,
        min_cfg_coef = 1.0,
        decoding_steps = [first_dec, 10, 10, 10],
        rescorer = rescorer,
        rescore_weights = 0.7,
        loop_trick_rotations = 0,
        loop_trick_aggregation = 'min_probs',
        loop_pad=0,
        seam_enforce=0,
        random_roll=False
    )

descriptions = [
    #'80s electronic track with melodic synthesizers, catchy beat and groovy bass',
    #'Techno guitar riff, powerful drum beat, 120 bpm',
    #'Ancient sumerian song with drums and string instruments',
    #'Medieval pipes marching orders',
    #'Anime intro with piano',
    #'Eurobeat track',
    #'Drumkit solo',
    #'Pirate music with accordion'
    'Drumkit solo with high hat kicks'
]
with torch.autocast(device_type=device, dtype=torch.float16):
    wav = model.generate(descriptions)

for i, (desc, one_wav) in enumerate(zip(descriptions, wav)):
    name = f"{i:02d}"
    name += "_loop" if loop_setup else ""
    name += f"_{seed}"
    name += f"_{desc}"

    # Will save under {idx}.wav, with loudness normalization at -16 db LUFS.
    audio_write(name, one_wav.to(device="cpu", dtype=torch.float32), model.sample_rate, strategy="loudness", loudness_headroom_db=18)