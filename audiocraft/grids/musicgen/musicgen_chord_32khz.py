# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from ._explorers import LMExplorer
from ...environment import AudioCraftEnvironment


@LMExplorer
def explorer(launcher):
    partitions = AudioCraftEnvironment.get_slurm_partitions(['team', 'global'])
    launcher.slurm_(gpus=8, partition=partitions)
    launcher.bind_(solver='musicgen/musicgen_chord_32khz')
    # replace this by the desired music dataset
    launcher.bind_(dset='chord_32khz')

    fsdp = {'autocast': False, 'fsdp.use': True}
    medium = {'model/lm/model_scale': 'medium'}
    large = {'model/lm/model_scale': 'large'}

    cfg_low = {'classifier_free_guidance.training_dropout': 0.2}
    wd_low = {'conditioners.description.t5.word_dropout': 0.2}

    adam = {'optim.optimizer': 'adamw', 'optim.lr': 1e-7}

    cache_path = {'conditioners.self_wav.chroma_chord.cache_path':
                  '/mnt/nvme/MSD_INST_vocalless/experiments/audiocraft/caches/chroma_chord'}

    # CACHE GENERATION JOBS
    n_cache_gen_jobs = 8
    gen_sub = launcher.slurm(gpus=8)
    gen_sub.bind_(
        cache_path, {
            # the cache is always computed over the whole file, so duration doesn't matter here.
            'dataset.segment_duration': 30.04084375,
            'dataset.batch_size': 16,
            'dataset.train.permutation_on_files': True,  # try to not repeat files.
            'optim.epochs': 1,
            'model/lm/model_scale': 'xsmall',
        })
    with gen_sub.job_array():
        for gen_job in range(n_cache_gen_jobs):
            gen_sub({'dataset.train.shuffle_seed': gen_job})

    # ACTUAL TRAINING JOBS.
    launcher.bind_(fsdp)

    launcher.slurm_(gpus=8).bind_(label='8gpus')
    with launcher.job_array():
        sub = launcher.bind()
        sub()
        # sub(cache_path)
        sub(medium, adam)

'''
    launcher.slurm_(gpus=32).bind_(label='32gpus')
    with launcher.job_array():
        sub = launcher.bind()
        sub()
        sub(cache_path)

    launcher.slurm_(gpus=64).bind_(label='64gpus')
    with launcher.job_array():
        sub = launcher.bind()
        sub(medium, adam)

    launcher.slurm_(gpus=96).bind_(label='96gpus')
    with launcher.job_array():
        sub = launcher.bind()
        sub(large, cfg_low, wd_low, adam, {'optim.max_norm': 3})
'''