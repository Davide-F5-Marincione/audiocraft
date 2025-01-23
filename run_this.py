import torch
import torchaudio
import numpy as np
import sys
import gc
from audiocraft.models import MusicGen, MAGNeT
from audiocraft.data.audio import audio_write
import tqdm

seed = 67
sec30 = False
loop_size_perc = .9
loops = [True]
k_loops = 2
rescore_weights = 0.0
prompt_original_length = 5.0
prompt_input_length = 2.0
use_rescorer = False
decoding_steps_per_tens = [130, 60, 25, 25]

descrs = [
    [
        "Genre Fusion, Jazz meets electronic dance music with a smooth saxophone lead",
        "Epic Trailer A cinematic score with powerful strings and booming percussion",
        "Funky Groove, A lively funk-inspired track with slap bass and vibrant horns",
        "Lullaby, A gentle and soothing melody with soft piano and harp"
    ]#,[
    #     "Retro Video Game, An 8-bit chiptune with a fast-paced and playful vibe",
    #     "Tropical Beach, A relaxing track with steel drums and marimba",
    #     "Dark Synthwave, A moody track with deep synth basslines and a cyberpunk edge",
    #     "Upbeat Pop, A catchy pop tune with vibrant synths and claps"
    # ],[
    #     "Medieval Fantasy A mystical track with lute, flute, and light percussion",
    #     "Ambient Chill, A serene, atmospheric piece with soft piano and pads",
    #     "Space Adventure, A futuristic track with ethereal synths and pulsating rhythms",
    #     "Energetic Hip-Hop, A beat featuring punchy drums and a dynamic melody"
    # ],[
    #     "Romantic Waltz, A flowing piano melody and warm strings, evoking charm",
    #     "High-Energy Rock, A guitar-driven anthem with powerful riffs",
    #     "Urban Groove, A jazzy lo-fi track with a steady beat and warm keys",
    #     "Holiday Cheer, A festive tune with sleigh bells and chimes"
    # ],[
    #     "Mystery and Suspense, A tense track with staccato strings and eerie effects",
    #     "Wild West, A cowboy-inspired track with twangy guitars and a galloping rhythm",
    #     "Spiritual Meditation, A calming track with Tibetan singing bowls and drones",
    #     "Fast-Paced Techno, A high-energy electronic track with hypnotic synth loops"
    # ],
    # [
    #     "Island Dance Party, A vibrant Caribbean soca-inspired track with rhythmic steel drums, upbeat maracas, and a groovy bassline",
    #     "Reggae Chillout, A laid-back reggae groove with a syncopated guitar skank, deep bass, and soft congas for a beachside vibe"
    # ]
]

assert torch.cuda.is_available(), "Man, you NEED a GPU for this"
device = "cuda"
rescorer = MusicGen.get_pretrained('facebook/musicgen-medium', device=device)
if sec30:
    model = MAGNeT.get_pretrained('facebook/magnet-medium-30secs', device=device)
else:
    model = MAGNeT.get_pretrained('facebook/magnet-medium-10secs', device=device)

for descriptions in tqdm.tqdm(descrs):
    for loop_setup in loops:
        torch.manual_seed(seed)
        np.random.seed(seed)

        rescorer.set_generation_params(
            duration = prompt_original_length * (3 if sec30 else 1)
        )

        if loop_setup:
            model.set_generation_params(
                span_arrangement = 'stride1',
                use_sampling = True,
                top_k=0,
                top_p = .9,
                temperature = 3.0,
                max_cfg_coef = 10.0,
                min_cfg_coef = 1.0,
                decoding_steps_per_tens = decoding_steps_per_tens,
                rescorer = rescorer.lm if use_rescorer else None,
                rescore_weights = rescore_weights,
                loop_size_perc = loop_size_perc,
                k_loops = k_loops
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
                decoding_steps_per_tens = [130, 60, 25, 25],
                rescorer = rescorer.lm if use_rescorer else None,
                rescore_weights = rescore_weights,
                loop_size_perc = 0,
                k_loops = k_loops
            )

        # gc.collect()

        with torch.autocast(device_type=device, dtype=torch.float16):
            prompt = rescorer.generate(descriptions)
            for i, (desc, one_wav) in enumerate(zip(descriptions, prompt)):
                name = f"samples/{seed}"
                name += "_prompt"
                name += f"_{desc}"

                # Will save under {idx}.wav, with loudness normalization at -16 db LUFS.
                audio_write(name, one_wav.to(device="cpu", dtype=torch.float32), rescorer.sample_rate, loudness_compressor=True, strategy="loudness", loudness_headroom_db=16)

            wav = model.generate_continuation(descriptions=descriptions, prompt=prompt[..., :int(prompt_input_length*rescorer.sample_rate)], prompt_sample_rate=rescorer.sample_rate)
            # wav = model.generate(descriptions=descriptions)

        sr = model.sample_rate
        # del model
        # del rescorer
        gc.collect()
        torch.cuda.empty_cache()

        for i, (desc, one_wav) in enumerate(zip(descriptions, wav)):
            name = f"samples/{seed}"
            name += "_loop" if loop_setup else ""
            name += f"_{desc}"

            # Will save under {idx}.wav, with loudness normalization at -16 db LUFS.
            audio_write(name, one_wav.to(device="cpu", dtype=torch.float32), sr, loudness_compressor=True, strategy="loudness", loudness_headroom_db=16)