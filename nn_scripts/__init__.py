"""
nn_scripts - neural audio fingerprinting package used by neffex_nn.py.

Modules:
    config         paths, hyperparameters, device selection, small helpers
    audio_library  MP3 decoding + 8 kHz int16 memmap cache (AudioLibrary)
    model          LogMel front-end, FingerprintNet CNN, NT-Xent loss, load_model
    augment        speaker-to-mic degradations for training and testing
    training       contrastive training loop
    index          window embedding + NNIndex
    matching       nearest-neighbour search + time-offset voting (Match, identify)
    evaluation     selftest and compare
"""
