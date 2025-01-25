import torch
import torchaudio
import numpy as np
import sys
import gc
from audiocraft.models import MusicGen, MAGNeT
from audiocraft.data.audio import audio_write
import tqdm

seed = 42
decoding_steps = [100, 50, 10, 10]

assert torch.cuda.is_available(), "Man, you NEED a GPU for this"
device = "cuda"
musicgen = MusicGen.get_pretrained('facebook/musicgen-medium', device=device)
magnet = MAGNeT.get_pretrained('facebook/magnet-medium-10secs', device=device)

descrs = [
    [
    ("jazzfusion","Genre Fusion, Jazz meets electronic dance music with a smooth saxophone lead"),
    ('epictrailer', "Epic Trailer A cinematic score with powerful strings and booming percussion"),
    ('funkygroove', "Funky Groove, A lively funk-inspired track with slap bass and vibrant horns"),
    ('lullaby', "Lullaby, A gentle and soothing melody with soft piano and harp")
    ], [
    ('retrogame', "Retro Video Game, An 8-bit chiptune with a fast-paced and playful vibe"),
    ('tropical', "Tropical Beach, A relaxing track with steel drums and marimba"),
    ('upbeatpop', "Upbeat Pop, A catchy pop tune with vibrant synths and claps"),
    ('medieval', "Medieval Fantasy A mystical track with lute, flute, and light percussion")
    ], [
    ('energetic', "Energetic Hip-Hop, A beat featuring punchy drums and a dynamic melody"),
    ('wildwest', "Wild West, A cowboy-inspired track with twangy guitars and a galloping rhythm"),
    ('waltz', "Romantic Waltz, A flowing piano melody and warm strings, evoking charm"),
    ('islandpary', "Island Dance Party, A vibrant Caribbean soca-inspired track with rhythmic steel drums, upbeat maracas, and a groovy bassline")
    ]
]

setups = [
    # ('magnet', -1,
    # {
    #     'span_arrangement' : 'stride1',
    #     'use_sampling' : True,
    #     'top_k' : 0,
    #     'top_p' : .9,
    #     'temperature' : 3.0,
    #     'max_cfg_coef' : 10.0,
    #     'min_cfg_coef' : 1.0,
    #     'decoding_steps' : decoding_steps,
    #     'rescorer' : None,
    #     'rescore_weights' : 0.0,
    #     'loop_size_perc' : 1.0,
    #     'k_loops' : 2
    # }),
    # ('hybrid_2s', 2.0,
    # {
    #     'span_arrangement' : 'stride1',
    #     'use_sampling' : True,
    #     'top_k' : 0,
    #     'top_p' : .9,
    #     'temperature' : 3.0,
    #     'max_cfg_coef' : 10.0,
    #     'min_cfg_coef' : 1.0,
    #     'decoding_steps' : decoding_steps,
    #     'rescorer' : None,
    #     'rescore_weights' : 0.0,
    #     'loop_size_perc' : 1.0,
    #     'k_loops' : 2
    # }),
    # ('hybrid_5s', 5.0,
    # {
    #     'span_arrangement' : 'stride1',
    #     'use_sampling' : True,
    #     'top_k' : 0,
    #     'top_p' : .9,
    #     'temperature' : 3.0,
    #     'max_cfg_coef' : 10.0,
    #     'min_cfg_coef' : 1.0,
    #     'decoding_steps' : decoding_steps,
    #     'rescorer' : None,
    #     'rescore_weights' : 0.0,
    #     'loop_size_perc' : 1.0,
    #     'k_loops' : 2
    # }),
    # ('magnet_circular', -1,
    # {
    #     'span_arrangement' : 'stride1',
    #     'use_sampling' : True,
    #     'top_k' : 0,
    #     'top_p' : .9,
    #     'temperature' : 3.0,
    #     'max_cfg_coef' : 10.0,
    #     'min_cfg_coef' : 1.0,
    #     'decoding_steps' : decoding_steps,
    #     'rescorer' : None,
    #     'rescore_weights' : 0.0,
    #     'loop_size_perc' : .8,
    #     'k_loops' : 2
    # }),
    # ('hybrid_2s_circular', 2.0,
    # {
    #     'span_arrangement' : 'stride1',
    #     'use_sampling' : True,
    #     'top_k' : 0,
    #     'top_p' : .9,
    #     'temperature' : 3.0,
    #     'max_cfg_coef' : 10.0,
    #     'min_cfg_coef' : 1.0,
    #     'decoding_steps' : decoding_steps,
    #     'rescorer' : None,
    #     'rescore_weights' : 0.0,
    #     'loop_size_perc' : .8,
    #     'k_loops' : 2
    # }),
    # ('hybrid_5s_circular', 5.0,
    # {
    #     'span_arrangement' : 'stride1',
    #     'use_sampling' : True,
    #     'top_k' : 0,
    #     'top_p' : .9,
    #     'temperature' : 3.0,
    #     'max_cfg_coef' : 10.0,
    #     'min_cfg_coef' : 1.0,
    #     'decoding_steps' : decoding_steps,
    #     'rescorer' : None,
    #     'rescore_weights' : 0.0,
    #     'loop_size_perc' : .8,
    #     'k_loops' : 2
    # }),
    # ('magnet_rescored_04', -1,
    # {
    #     'span_arrangement' : 'stride1',
    #     'use_sampling' : True,
    #     'top_k' : 0,
    #     'top_p' : .9,
    #     'temperature' : 3.0,
    #     'max_cfg_coef' : 10.0,
    #     'min_cfg_coef' : 1.0,
    #     'decoding_steps' : decoding_steps,
    #     'rescorer' : musicgen.lm,
    #     'rescore_weights' : 0.4,
    #     'loop_size_perc' : 1.0,
    #     'k_loops' : 2
    # }),
    ('hybrid_2s_rescored_04', 2.0,
    {
        'span_arrangement' : 'stride1',
        'use_sampling' : True,
        'top_k' : 0,
        'top_p' : .9,
        'temperature' : 3.0,
        'max_cfg_coef' : 10.0,
        'min_cfg_coef' : 1.0,
        'decoding_steps' : decoding_steps,
        'rescorer' : musicgen.lm,
        'rescore_weights' : 0.4,
        'loop_size_perc' : 1.0,
        'k_loops' : 2
    }),
    # ('hybrid_5s_rescored_04', 5.0,
    # {
    #     'span_arrangement' : 'stride1',
    #     'use_sampling' : True,
    #     'top_k' : 0,
    #     'top_p' : .9,
    #     'temperature' : 3.0,
    #     'max_cfg_coef' : 10.0,
    #     'min_cfg_coef' : 1.0,
    #     'decoding_steps' : decoding_steps,
    #     'rescorer' : musicgen.lm,
    #     'rescore_weights' : 0.4,
    #     'loop_size_perc' : 1.0,
    #     'k_loops' : 2
    # }),
    # ('magnet_circular_rescored_04', -1,
    # {
    #     'span_arrangement' : 'stride1',
    #     'use_sampling' : True,
    #     'top_k' : 0,
    #     'top_p' : .9,
    #     'temperature' : 3.0,
    #     'max_cfg_coef' : 10.0,
    #     'min_cfg_coef' : 1.0,
    #     'decoding_steps' : decoding_steps,
    #     'rescorer' : musicgen.lm,
    #     'rescore_weights' : 0.4,
    #     'loop_size_perc' : .8,
    #     'k_loops' : 2
    # }),
    # ('hybrid_2s_circular_rescored_04', 2.0,
    # {
    #     'span_arrangement' : 'stride1',
    #     'use_sampling' : True,
    #     'top_k' : 0,
    #     'top_p' : .9,
    #     'temperature' : 3.0,
    #     'max_cfg_coef' : 10.0,
    #     'min_cfg_coef' : 1.0,
    #     'decoding_steps' : decoding_steps,
    #     'rescorer' : musicgen.lm,
    #     'rescore_weights' : 0.4,
    #     'loop_size_perc' : .8,
    #     'k_loops' : 2
    # }),
    # ('hybrid_5s_circular_rescored_04', 5.0,
    # {
    #     'span_arrangement' : 'stride1',
    #     'use_sampling' : True,
    #     'top_k' : 0,
    #     'top_p' : .9,
    #     'temperature' : 3.0,
    #     'max_cfg_coef' : 10.0,
    #     'min_cfg_coef' : 1.0,
    #     'decoding_steps' : decoding_steps,
    #     'rescorer' : musicgen.lm,
    #     'rescore_weights' : 0.4,
    #     'loop_size_perc' : .8,
    #     'k_loops' : 2
    # }),
    # ('magnet_rescored_08', -1,
    # {
    #     'span_arrangement' : 'stride1',
    #     'use_sampling' : True,
    #     'top_k' : 0,
    #     'top_p' : .9,
    #     'temperature' : 3.0,
    #     'max_cfg_coef' : 10.0,
    #     'min_cfg_coef' : 1.0,
    #     'decoding_steps' : decoding_steps,
    #     'rescorer' : musicgen.lm,
    #     'rescore_weights' : 0.8,
    #     'loop_size_perc' : 1.0,
    #     'k_loops' : 2
    # }),
    ('hybrid_2s_rescored_08', 2.0,
    {
        'span_arrangement' : 'stride1',
        'use_sampling' : True,
        'top_k' : 0,
        'top_p' : .9,
        'temperature' : 3.0,
        'max_cfg_coef' : 10.0,
        'min_cfg_coef' : 1.0,
        'decoding_steps' : decoding_steps,
        'rescorer' : musicgen.lm,
        'rescore_weights' : 0.8,
        'loop_size_perc' : 1.0,
        'k_loops' : 2
    }),
    ('hybrid_5s_rescored_08', 5.0,
    {
        'span_arrangement' : 'stride1',
        'use_sampling' : True,
        'top_k' : 0,
        'top_p' : .9,
        'temperature' : 3.0,
        'max_cfg_coef' : 10.0,
        'min_cfg_coef' : 1.0,
        'decoding_steps' : decoding_steps,
        'rescorer' : musicgen.lm,
        'rescore_weights' : 0.8,
        'loop_size_perc' : 1.0,
        'k_loops' : 2
    }),
    # ('magnet_circular_rescored_08', -1,
    # {
    #     'span_arrangement' : 'stride1',
    #     'use_sampling' : True,
    #     'top_k' : 0,
    #     'top_p' : .9,
    #     'temperature' : 3.0,
    #     'max_cfg_coef' : 10.0,
    #     'min_cfg_coef' : 1.0,
    #     'decoding_steps' : decoding_steps,
    #     'rescorer' : musicgen.lm,
    #     'rescore_weights' : 0.8,
    #     'loop_size_perc' : 0.8,
    #     'k_loops' : 2
    # }),
    # ('hybrid_2s_circular_rescored_08', 2.0,
    # {
    #     'span_arrangement' : 'stride1',
    #     'use_sampling' : True,
    #     'top_k' : 0,
    #     'top_p' : .9,
    #     'temperature' : 3.0,
    #     'max_cfg_coef' : 10.0,
    #     'min_cfg_coef' : 1.0,
    #     'decoding_steps' : decoding_steps,
    #     'rescorer' : musicgen.lm,
    #     'rescore_weights' : 0.8,
    #     'loop_size_perc' : .8,
    #     'k_loops' : 2
    # }),
    # ('hybrid_5s_circular_rescored_08', 5.0,
    # {
    #     'span_arrangement' : 'stride1',
    #     'use_sampling' : True,
    #     'top_k' : 0,
    #     'top_p' : .9,
    #     'temperature' : 3.0,
    #     'max_cfg_coef' : 10.0,
    #     'min_cfg_coef' : 1.0,
    #     'decoding_steps' : decoding_steps,
    #     'rescorer' : musicgen.lm,
    #     'rescore_weights' : 0.8,
    #     'loop_size_perc' : .8,
    #     'k_loops' : 2
    # }),
    # ('magnet_rescored_1', -1,
    # {
    #     'span_arrangement' : 'stride1',
    #     'use_sampling' : True,
    #     'top_k' : 0,
    #     'top_p' : .9,
    #     'temperature' : 3.0,
    #     'max_cfg_coef' : 10.0,
    #     'min_cfg_coef' : 1.0,
    #     'decoding_steps' : decoding_steps,
    #     'rescorer' : musicgen.lm,
    #     'rescore_weights' : 1.0,
    #     'loop_size_perc' : 1.0,
    #     'k_loops' : 2
    # }),
    ('hybrid_2s_rescored_1', 2.0,
    {
        'span_arrangement' : 'stride1',
        'use_sampling' : True,
        'top_k' : 0,
        'top_p' : .9,
        'temperature' : 3.0,
        'max_cfg_coef' : 10.0,
        'min_cfg_coef' : 1.0,
        'decoding_steps' : decoding_steps,
        'rescorer' : musicgen.lm,
        'rescore_weights' : 1.0,
        'loop_size_perc' : 1.0,
        'k_loops' : 2
    }),
    ('hybrid_5s_rescored_1', 5.0,
    {
        'span_arrangement' : 'stride1',
        'use_sampling' : True,
        'top_k' : 0,
        'top_p' : .9,
        'temperature' : 3.0,
        'max_cfg_coef' : 10.0,
        'min_cfg_coef' : 1.0,
        'decoding_steps' : decoding_steps,
        'rescorer' : musicgen.lm,
        'rescore_weights' : 1.0,
        'loop_size_perc' : 1.0,
        'k_loops' : 2
    }),
    # ('magnet_circular_rescored_1', -1,
    # {
    #     'span_arrangement' : 'stride1',
    #     'use_sampling' : True,
    #     'top_k' : 0,
    #     'top_p' : .9,
    #     'temperature' : 3.0,
    #     'max_cfg_coef' : 10.0,
    #     'min_cfg_coef' : 1.0,
    #     'decoding_steps' : decoding_steps,
    #     'rescorer' : musicgen.lm,
    #     'rescore_weights' : 1.0,
    #     'loop_size_perc' : .8,
    #     'k_loops' : 2
    # }),
    # ('hybrid_2s_circular_rescored_1', 2.0,
    # {
    #     'span_arrangement' : 'stride1',
    #     'use_sampling' : True,
    #     'top_k' : 0,
    #     'top_p' : .9,
    #     'temperature' : 3.0,
    #     'max_cfg_coef' : 10.0,
    #     'min_cfg_coef' : 1.0,
    #     'decoding_steps' : decoding_steps,
    #     'rescorer' : musicgen.lm,
    #     'rescore_weights' : 1.0,
    #     'loop_size_perc' : .8,
    #     'k_loops' : 2
    # }),
    # ('hybrid_5s_circular_rescored_1', 5.0,
    # {
    #     'span_arrangement' : 'stride1',
    #     'use_sampling' : True,
    #     'top_k' : 0,
    #     'top_p' : .9,
    #     'temperature' : 3.0,
    #     'max_cfg_coef' : 10.0,
    #     'min_cfg_coef' : 1.0,
    #     'decoding_steps' : decoding_steps,
    #     'rescorer' : musicgen.lm,
    #     'rescore_weights' : 1.0,
    #     'loop_size_perc' : .8,
    #     'k_loops' : 2
    # })
]

for descriptions in tqdm.tqdm(descrs):
    folders, descriptions = zip(*descriptions)
    torch.manual_seed(seed)
    np.random.seed(seed)

    musicgen.set_generation_params(
        duration = 10.0
    )

    prompt = []
    for folder, d in zip(folders, descriptions):
        name = f"samples/{folder}_{seed}/"
        name += "musicgen-medium"
        p = musicgen.generate([d])

        prompt.append(p)

        p = p[0].to(device="cpu", dtype=torch.float32)

        audio_write(name, torch.cat([p, p], -1), musicgen.sample_rate, loudness_compressor=True, strategy="loudness", loudness_headroom_db=16)

    prompt = torch.cat(prompt, 0).contiguous()
    gc.collect()
    torch.cuda.empty_cache()

    for filename, promptlen, setup in setups:
        torch.manual_seed(seed)
        np.random.seed(seed)

        magnet.set_generation_params(
            **setup
        )

        with torch.autocast(device_type=device, dtype=torch.float16):
            if promptlen > 0:
                results = magnet.generate_continuation(descriptions=descriptions, prompt=prompt[..., :int(promptlen*musicgen.sample_rate)], prompt_sample_rate=musicgen.sample_rate)
            else:
                results = magnet.generate(descriptions=descriptions)

        gc.collect()
        torch.cuda.empty_cache()

        for folder, wav in zip(folders, results):
            name = f"samples/{folder}_{seed}/"
            name += filename

            audio_write(name, wav.to(device="cpu", dtype=torch.float32), magnet.sample_rate, loudness_compressor=True, strategy="loudness", loudness_headroom_db=16)