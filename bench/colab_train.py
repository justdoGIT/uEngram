"""Label-free CNN2 training on an explicitly uploaded public sensor archive.

Local: prepare(load_stream(name), '/path/input.npz'). Upload this module,
encoding_cache.py, encoder_gate.py and uengram/{encoder_cnn2,encoders,stats}.py
into matching /content packages. In Colab call train(input_path, output_dir).
No authentication, purchases, or dataset downloading occurs in this module.
"""
from __future__ import annotations

import copy
import json
import os
import pickle
from types import SimpleNamespace

import numpy as np

from bench.encoding_cache import input_hash
from bench.encoder_gate import accept_gate, gate_stats


def prepare(stream, path):
    arrays = dict(W=np.asarray(stream.W, np.float32), fit=stream.fit, cal=stream.cal,
                  segment=stream.segment, end=stream.end.astype(np.int64), name=stream.name)
    if stream.cond is not None:
        arrays['cond'] = stream.cond.astype(np.float32)
    np.savez_compressed(path, **arrays)


def train(input_path, output_dir, seeds=range(5), conditioned=False):
    # Must precede CUDA initialization for deterministic CUDA matrix products.
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    from uengram.encoder_cnn2 import CNN2Encoder
    with np.load(input_path) as z:
        s = SimpleNamespace(W=z['W'], fit=z['fit'], cal=z['cal'], segment=z['segment'],
                            end=z['end'].astype('datetime64[s]'), name=str(z['name']),
                            cond=z['cond'] if conditioned else None)
    name = 'cnn2s' if conditioned else 'cnn2'
    fingerprint = input_hash(s, name)
    os.makedirs(output_dir, exist_ok=True)
    records = []
    for seed in seeds:
        record = dict(stream=s.name, encoder=name, seed=int(seed), input_hash=fingerprint,
                      torch_version=torch.__version__, numpy_version=np.__version__)
        try:
            fit_kw = {'cond': s.cond[s.fit]} if conditioned else {}
            enc = CNN2Encoder(seed=int(seed)).fit(s.W[s.fit], 1, s.segment[s.fit], **fit_kw)
            stats = gate_stats(enc, s.W[s.fit], s.W[s.cal], s.segment[s.fit],
                               s.cond[s.fit] if conditioned else None,
                               s.cond[s.cal] if conditioned else None)
            record.update(stats, accepted=accept_gate(stats), fit_seconds=enc.fit_seconds,
                          backend=str(enc.device), medoid=int(enc.medoid))
            if record['accepted']:
                kw = {'cond': s.cond} if conditioned else {}
                Z = enc.encode(s.W, **kw)
                stem = os.path.join(output_dir, f'{s.name}_{name}_{seed}')
                np.savez_compressed(stem + '.npz', Z=Z, sigma0=enc.sigma0,
                                    metadata=json.dumps(record))
                # Trusted local checkpoint, CPU-portable. Never unpickle files
                # supplied by an untrusted party.
                saved = copy.deepcopy(enc)
                saved.enc = saved.enc.to('cpu')
                saved.device = torch.device('cpu')
                with open(stem + '.pkl', 'wb') as file:
                    pickle.dump(saved, file, protocol=pickle.HIGHEST_PROTOCOL)
        except (ValueError, np.linalg.LinAlgError) as error:
            record.update(accepted=False, failure=f'{type(error).__name__}: {error}')
        records.append(record)
        print(json.dumps(record), flush=True)
        with open(os.path.join(output_dir, f'{s.name}_{name}_gate.json'), 'w') as file:
            json.dump(records, file, indent=2)
    return records
