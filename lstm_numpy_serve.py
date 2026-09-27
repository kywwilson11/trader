"""Numpy-only replica of ``model_v2.RegressionLSTM.forward`` (eval mode).

RESEARCH / MEASUREMENT ONLY (ENGINE X7, research/campaign_2026-09_jetson/
research_engine.md section E5 + "X7 spike results"). Nothing on the live path
imports this module: ``predict_now`` still serves the torch model, and
``tests/test_lstm_numpy_serve.py`` pins that predict_now does NOT import it.
Wiring it in is an owner flip (proposed flag ``TRADER_NUMPY_LSTM_SERVE``,
default OFF) — see the X7 appendix for the design and the measured evidence.

Why: on the Jetson ``import torch`` costs ~330 MB RSS per bot process while
the champion model is ~0.3-1.3 M float32 parameters. Evaluating the SAME
trained ``state_dict`` with numpy removes the torch import from the serving
process without touching the model, the features or training (training stays
on torch; this module only reads its weights).

Import contract: numpy + stdlib only at import time. torch is imported lazily
inside ``export_from_pth`` (and the ``__main__`` parity harness) only.

Exact op mapping (torch eval-mode op -> numpy op here). Every weight is used
exactly as stored in the state_dict (no re-fitting, no folding except the two
LSTM biases, which torch also simply adds):

  model_v2.RegressionLSTM.forward(x)            NumpyLSTM.forward(x), x (B,T,F)
  --------------------------------------------  -----------------------------------------------
  nn.LSTM(batch_first, num_layers=L, h0=c0=0)   per layer l (unidirectional, no proj_size):
    gates = x_t W_ih^T + b_ih + h W_hh^T + b_hh   xi = x @ W_ih_l.T + (b_ih_l + b_hh_l)  (all t)
                                                  g  = xi[:, t] + h @ W_hh_l.T
    chunk order i, f, g, o (torch doc)            i,f,g,o = g[:, 0:H], [H:2H], [2H:3H], [3H:4H]
    i,f,o = sigmoid(.); g = tanh(.)               sigmoid(z) = 1/(1+exp(-z)); np.tanh
    c = f*c + i*g ; h = o*tanh(c)                 identical
    inter-layer dropout (eval) = identity         omitted
  nn.MultiheadAttention(E=H, n_heads, bf=True)  self-attention, q = k = v = lstm_out:
    in_proj: [q|k|v] = X W_in^T + b_in            qkv = X @ in_proj_weight.T + in_proj_bias
    heads: split E into n_heads x hd (hd=E/nh)    reshape (B,T,nh,hd) -> (B,nh,T,hd)
    q * sqrt(1/hd) ; softmax(q k^T) over keys     q * sqrt(1/hd); stable softmax (max-shift)
    attn dropout (eval) = identity                omitted
    concat heads ; out_proj                       (B,T,E) @ out_proj.weight.T + out_proj.bias
    (returned attention weights are discarded)    not computed
  residual + nn.LayerNorm(H, eps)               z = a + X; mu = mean_H(z);
                                                  var = mean_H((z-mu)^2) (biased, as torch)
                                                  (z-mu) * (1/sqrt(var+eps)) * w + b
  combined.mean(dim=1)                          mean over T (axis 1)
  fc: Linear -> ReLU -> Dropout -> Linear       relu(p @ W0.T + b0) @ W3.T + b3
    Dropout (eval) = identity                     omitted
  .squeeze(-1)                                  [:, 0] -> (B,)

Fail-closed export: ``export_weights`` accepts ONLY the exact key set above
(unidirectional, no projections, no attention bias_k/bias_v, packed
in_proj). Any architecture change in model_v2 therefore breaks the export
loudly instead of silently mis-serving.

Scope NOT replicated (stays in predict_now either way): feature computation,
the non-finite guard, ``scaler_X.transform`` (sklearn RobustScaler), the
LightGBM / q10 legs and the blend. The numpy model consumes exactly the
scaled (1, seq_len, n_features) window predict_now hands the torch model.
"""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np

FORMAT_VERSION = 1
ARCH = 'RegressionLSTM'
_META_KEY = '__meta__'


def _sigmoid(z):
    # torch CPU sigmoid is 1/(1+exp(-z)); exp overflow -> inf -> 0.0 exactly,
    # the same limit torch returns, so silence only the overflow warning.
    with np.errstate(over='ignore'):
        return 1.0 / (1.0 + np.exp(-z))


def _to_numpy(v):
    """Tensor (any device) or array -> numpy array, no torch import."""
    if hasattr(v, 'detach'):
        v = v.detach()
        if hasattr(v, 'cpu'):
            v = v.cpu()
        return v.numpy().copy()
    return np.array(v, copy=True)


def _expected_keys(num_layers):
    keys = set()
    for l in range(num_layers):
        keys |= {f'lstm.weight_ih_l{l}', f'lstm.weight_hh_l{l}',
                 f'lstm.bias_ih_l{l}', f'lstm.bias_hh_l{l}'}
    keys |= {'attn.in_proj_weight', 'attn.in_proj_bias',
             'attn.out_proj.weight', 'attn.out_proj.bias',
             'norm.weight', 'norm.bias',
             'fc.0.weight', 'fc.0.bias', 'fc.3.weight', 'fc.3.bias'}
    return keys


def _validate(sd, n_heads):
    """Check the key set and every shape; return (input_dim, H, L, M)."""
    n_layers = 0
    while f'lstm.weight_ih_l{n_layers}' in sd:
        n_layers += 1
    if n_layers == 0:
        raise ValueError('state_dict has no lstm.weight_ih_l0')
    exp = _expected_keys(n_layers)
    got = set(sd)
    if got != exp:
        raise ValueError(
            'unsupported RegressionLSTM state_dict (fail-closed): '
            f'extra={sorted(got - exp)} missing={sorted(exp - got)}')
    four_h, input_dim = sd['lstm.weight_ih_l0'].shape
    if four_h % 4:
        raise ValueError(f'lstm gate dim {four_h} not divisible by 4')
    H = four_h // 4
    for l in range(n_layers):
        want_in = input_dim if l == 0 else H
        _shape(sd, f'lstm.weight_ih_l{l}', (4 * H, want_in))
        _shape(sd, f'lstm.weight_hh_l{l}', (4 * H, H))
        _shape(sd, f'lstm.bias_ih_l{l}', (4 * H,))
        _shape(sd, f'lstm.bias_hh_l{l}', (4 * H,))
    _shape(sd, 'attn.in_proj_weight', (3 * H, H))
    _shape(sd, 'attn.in_proj_bias', (3 * H,))
    _shape(sd, 'attn.out_proj.weight', (H, H))
    _shape(sd, 'attn.out_proj.bias', (H,))
    _shape(sd, 'norm.weight', (H,))
    _shape(sd, 'norm.bias', (H,))
    M = sd['fc.0.weight'].shape[0]
    _shape(sd, 'fc.0.weight', (M, H))
    _shape(sd, 'fc.0.bias', (M,))
    _shape(sd, 'fc.3.weight', (1, M))
    _shape(sd, 'fc.3.bias', (1,))
    n_heads = int(n_heads)
    if n_heads < 1 or H % n_heads:
        raise ValueError(f'hidden_dim {H} not divisible by n_heads {n_heads}')
    return int(input_dim), int(H), int(n_layers), int(M)


def _shape(sd, k, want):
    if tuple(sd[k].shape) != tuple(want):
        raise ValueError(f'{k}: shape {tuple(sd[k].shape)} != {tuple(want)}')


def sha256_file(path, _chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(_chunk), b''):
            h.update(block)
    return h.hexdigest()


def export_weights(state_dict_or_model, path, n_heads=None, ln_eps=None,
                   source_pth=None):
    """Write every tensor RegressionLSTM.forward needs to a ``.npz``.

    state_dict_or_model: a RegressionLSTM (eager or jit-traced) or its
    state_dict (tensors or arrays). n_heads is not recoverable from weight
    shapes: pass it (config['n_heads']) unless a model with ``.attn.num_heads``
    is given. ln_eps defaults to the model's ``norm.eps`` or torch's 1e-5.
    source_pth (optional) records the .pth sha256 so ``is_fresh`` can prove
    the npz matches the weights file. Weights keep their stored dtype
    (float32 for every trained champion); the write is atomic.
    Returns the metadata dict.
    """
    obj = state_dict_or_model
    if hasattr(obj, 'state_dict') and callable(obj.state_dict):
        if n_heads is None:
            n_heads = getattr(getattr(obj, 'attn', None), 'num_heads', None)
        if ln_eps is None:
            ln_eps = getattr(getattr(obj, 'norm', None), 'eps', None)
        obj = obj.state_dict()
    if n_heads is None:
        raise ValueError('n_heads is required when exporting a bare state_dict')
    sd = {k: _to_numpy(v) for k, v in obj.items()}
    input_dim, H, L, M = _validate(sd, n_heads)
    meta = {
        'format_version': FORMAT_VERSION, 'arch': ARCH,
        'input_dim': input_dim, 'hidden_dim': H, 'num_layers': L,
        'fc_hidden': M, 'n_heads': int(n_heads),
        'ln_eps': float(1e-5 if ln_eps is None else ln_eps),
        'weight_dtype': str(sd['lstm.weight_ih_l0'].dtype),
    }
    if source_pth is not None:
        meta['source_pth'] = os.path.basename(str(source_pth))
        meta['source_sha256'] = sha256_file(source_pth)
    arrays = dict(sd)
    arrays[_META_KEY] = np.array(json.dumps(meta, sort_keys=True))
    path = str(path)
    tmp = f'{path}.tmp.{os.getpid()}'
    try:
        with open(tmp, 'wb') as f:
            np.savez(f, **arrays)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return meta


def export_from_pth(pth_path, out_path, n_heads, ln_eps=None):
    """Export a saved ``{prefix}model_v2.pth`` (torch imported lazily here)."""
    import torch
    sd = torch.load(pth_path, map_location='cpu', weights_only=True)
    return export_weights(sd, out_path, n_heads=n_heads, ln_eps=ln_eps,
                          source_pth=pth_path)


def read_meta(npz_path):
    with np.load(npz_path, allow_pickle=False) as z:
        return json.loads(str(z[_META_KEY][()]))


def is_fresh(npz_path, pth_path):
    """True iff the npz exists and was exported from exactly this .pth."""
    try:
        meta = read_meta(npz_path)
        return meta.get('source_sha256') == sha256_file(pth_path)
    except (OSError, ValueError, KeyError):
        return False


class NumpyLSTM:
    """Eval-mode RegressionLSTM forward in numpy. See module docstring."""

    def __init__(self, weights, meta, dtype=np.float32):
        self.meta = dict(meta)
        self.dtype = np.dtype(dtype)
        if self.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise ValueError(f'dtype must be float32 or float64, got {dtype}')
        _validate(weights, self.meta['n_heads'])
        self.input_dim = int(self.meta['input_dim'])
        self.H = int(self.meta['hidden_dim'])
        self.L = int(self.meta['num_layers'])
        self.n_heads = int(self.meta['n_heads'])
        self.eps = float(self.meta['ln_eps'])
        dt = self.dtype

        def c(a):  # transposed, contiguous, target dtype (fp32->fp64 exact)
            return np.ascontiguousarray(np.asarray(a).astype(dt, copy=False))

        self._lstm = []
        for l in range(self.L):
            self._lstm.append((
                c(weights[f'lstm.weight_ih_l{l}'].T),
                c(weights[f'lstm.weight_hh_l{l}'].T),
                # torch adds b_ih and b_hh separately; summing them first
                # only reorders two float adds (rounding-level, see tests)
                c(weights[f'lstm.bias_ih_l{l}'].astype(dt)
                  + weights[f'lstm.bias_hh_l{l}'].astype(dt)),
            ))
        self._w_in = c(weights['attn.in_proj_weight'].T)
        self._b_in = c(weights['attn.in_proj_bias'])
        self._w_out = c(weights['attn.out_proj.weight'].T)
        self._b_out = c(weights['attn.out_proj.bias'])
        self._ln_w = c(weights['norm.weight'])
        self._ln_b = c(weights['norm.bias'])
        self._w0 = c(weights['fc.0.weight'].T)
        self._b0 = c(weights['fc.0.bias'])
        self._w3 = c(weights['fc.3.weight'].T)
        self._b3 = c(weights['fc.3.bias'])
        self._q_scale = dt.type(np.sqrt(1.0 / float(self.H // self.n_heads)))

    @classmethod
    def from_npz(cls, path, dtype=np.float32):
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z[_META_KEY][()]))
            if meta.get('arch') != ARCH or meta.get('format_version') != FORMAT_VERSION:
                raise ValueError(f'unsupported npz: arch={meta.get("arch")} '
                                 f'format={meta.get("format_version")}')
            weights = {k: z[k] for k in z.files if k != _META_KEY}
        return cls(weights, meta, dtype=dtype)

    def forward(self, x):
        """x: (B, T, input_dim) -> (B,) predictions in ``self.dtype``."""
        x = np.asarray(x)
        if x.ndim != 3 or x.shape[2] != self.input_dim:
            raise ValueError(f'expected (B, T, {self.input_dim}), got {x.shape}')
        dt = self.dtype
        seq = x.astype(dt, copy=False)
        B, T, _ = seq.shape
        H = self.H
        for w_ih_t, w_hh_t, b in self._lstm:
            xi = seq @ w_ih_t + b                      # (B, T, 4H)
            h = np.zeros((B, H), dt)
            cst = np.zeros((B, H), dt)
            out = np.empty((B, T, H), dt)
            for t in range(T):
                g = xi[:, t, :] + h @ w_hh_t
                i = _sigmoid(g[:, :H])
                f = _sigmoid(g[:, H:2 * H])
                gg = np.tanh(g[:, 2 * H:3 * H])
                o = _sigmoid(g[:, 3 * H:])
                cst = f * cst + i * gg
                h = o * np.tanh(cst)
                out[:, t, :] = h
            seq = out
        X = seq                                        # lstm_out (B, T, H)
        nh = self.n_heads
        hd = H // nh
        qkv = X @ self._w_in + self._b_in              # (B, T, 3H)

        def heads(a):
            return a.reshape(B, T, nh, hd).transpose(0, 2, 1, 3)

        q = heads(qkv[..., :H]) * self._q_scale
        k = heads(qkv[..., H:2 * H])
        v = heads(qkv[..., 2 * H:])
        s = q @ k.transpose(0, 1, 3, 2)                # (B, nh, T, T)
        s = np.exp(s - s.max(axis=-1, keepdims=True))
        s = s / s.sum(axis=-1, keepdims=True)
        a = (s @ v).transpose(0, 2, 1, 3).reshape(B, T, H)
        a = a @ self._w_out + self._b_out
        z = a + X
        mu = z.mean(axis=-1, keepdims=True)
        zc = z - mu
        var = (zc * zc).mean(axis=-1, keepdims=True)
        z = zc * (1.0 / np.sqrt(var + dt.type(self.eps))) * self._ln_w + self._ln_b
        pooled = z.mean(axis=1)                        # (B, H)
        h1 = np.maximum(pooled @ self._w0 + self._b0, 0)
        return (h1 @ self._w3 + self._b3)[:, 0]

    __call__ = forward


# ---------------------------------------------------------------------------
# __main__: X7 parity / memory harness on the REAL artifacts (Jetson only)
# ---------------------------------------------------------------------------

_REPO = os.path.dirname(os.path.abspath(__file__))


def _vmrss_mb():
    with open('/proc/self/status') as f:
        for line in f:
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) / 1024.0
    return float('nan')


def _paths(artifacts_dir, book):
    p = 'stock_' if book == 'stock' else ''
    d = os.path.abspath(artifacts_dir)
    return {k: os.path.join(d, f'{p}{v}') for k, v in (
        ('model', 'model_v2.pth'), ('config', 'config_v2.pkl'),
        ('scaler', 'scaler_v2.pkl'), ('features', 'feature_cols_v2.pkl'))}


def _cmd_extract(a):
    """Draw real scaled windows from the training store (no torch here)."""
    import sys
    sys.path.insert(0, _REPO)
    import joblib
    from data_utils import load_training_data
    P = _paths(a.artifacts_dir, a.book)
    config = joblib.load(P['config'])
    scaler = joblib.load(P['scaler'])
    fcols = list(joblib.load(P['features']))
    T = int(config['seq_len'])
    df = load_training_data(a.book, columns=fcols + ['Ticker'])
    rng = np.random.default_rng(a.seed)
    per_ticker = {}
    for tk, d in df.groupby('Ticker', sort=True):
        # predict_now: rows NaN in any consumed column are dropped, then the
        # LAST seq_len rows form the window (closed bars, time order).
        d = d.sort_index(kind='stable').dropna(subset=fcols)
        per_ticker[tk] = (d.index.asi8, d[fcols].to_numpy(np.float64))
    del df
    all_ts = np.unique(np.concatenate([v[0][T - 1:] for v in per_ticker.values()]))
    chosen = np.sort(rng.choice(all_ts, size=min(a.n_ts, len(all_ts)), replace=False))
    wins, ts_out, tk_out, n_nonfinite = [], [], [], 0
    for ts in chosen:
        for tk, (idx, vals) in per_ticker.items():
            pos = np.searchsorted(idx, ts)
            if pos >= len(idx) or idx[pos] != ts or pos < T - 1:
                continue
            raw = vals[pos - T + 1:pos + 1]
            if not np.isfinite(raw).all():
                n_nonfinite += 1
                continue
            # exactly predict_now's step: float64 transform of the last
            # seq_len rows (row-independent RobustScaler).
            wins.append(scaler.transform(raw))
            ts_out.append(ts)
            tk_out.append(tk)
    np.savez(a.out, windows=np.stack(wins), ts=np.asarray(ts_out, np.int64),
             tickers=np.asarray(tk_out), n_nonfinite=np.int64(n_nonfinite),
             peak_rss_mb=np.float64(_peak_rss_mb()))
    print(f'[extract] {a.book}: {len(wins)} windows over {len(chosen)} '
          f'timestamps, T={T}, F={len(fcols)}, skipped non-finite={n_nonfinite}, '
          f'peak RSS {_peak_rss_mb():.0f} MB')


def _peak_rss_mb():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _verdicts(p, th):
    """predict_now's three-way recommendation: +1 BUY, -1 SELL/AVOID, 0 HOLD."""
    return np.where(p > th, 1, np.where(p < -th, -1, 0))


def _ranks(p, groups):
    r = np.empty(len(p), np.int64)
    for g in np.unique(groups):
        m = np.flatnonzero(groups == g)
        r[m] = np.argsort(np.argsort(-p[m], kind='stable'), kind='stable')
    return r


def _cmd_parity(a):
    import subprocess
    import sys
    import tempfile
    import time
    sys.path.insert(0, _REPO)
    P = _paths(a.artifacts_dir, a.book)
    work = a.work_dir or tempfile.mkdtemp(prefix='x7_')
    os.makedirs(work, exist_ok=True)
    wpath = os.path.join(work, f'{a.book}_windows.npz')
    if not os.path.exists(wpath):
        # separate process: the parquet load's RSS never shares a process
        # with torch (Jetson 8 GB etiquette)
        subprocess.run([sys.executable, os.path.abspath(__file__), 'extract',
                        '--book', a.book, '--artifacts-dir', a.artifacts_dir,
                        '--n-ts', str(a.n_ts), '--seed', str(a.seed),
                        '--out', wpath], check=True)
    Z = np.load(wpath, allow_pickle=False)
    W64, groups, n = Z['windows'], Z['ts'], len(Z['ts'])
    x32 = W64.astype(np.float32)          # == torch.tensor(seq, float32)

    import torch
    import predict_now                    # the real serving loader
    from model_v2 import RegressionLSTM
    cwd = os.getcwd()
    os.chdir(os.path.dirname(P['model']))  # load_model paths are cwd-relative
    try:
        prefix = 'stock' if a.book == 'stock' else ''
        jit_model, _scaler, config, T, fcols = predict_now.load_model('cpu', prefix)
    finally:
        os.chdir(cwd)
    th = float(config.get('trade_threshold', 0.15))
    eager = RegressionLSTM(input_dim=config['input_dim'],
                           hidden_dim=config['hidden_dim'],
                           num_layers=config['num_layers'],
                           dropout=config['dropout'],
                           n_heads=config.get('n_heads', 4))
    eager.load_state_dict(torch.load(P['model'], map_location='cpu',
                                     weights_only=True))
    eager.eval()
    npz = os.path.join(work, f'{prefix + "_" if prefix else ""}model_v2.npz')
    meta = export_weights(eager, npz, source_pth=P['model'])
    assert meta['n_heads'] == config.get('n_heads', 4)
    np32 = NumpyLSTM.from_npz(npz, np.float32)
    np64 = NumpyLSTM.from_npz(npz, np.float64)

    def timed(fn, reps):
        fn()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        return (time.perf_counter() - t0) / reps * 1e3

    # Serving-literal torch: B=1 per window, jit model, float32, inference_mode
    t32 = np.empty(n)
    t0 = time.perf_counter()
    with torch.inference_mode():
        for j in range(n):
            t32[j] = float(jit_model(torch.tensor(W64[j].reshape(1, T, -1),
                                                  dtype=torch.float32)).cpu().item())
    ms_jit = (time.perf_counter() - t0) / n * 1e3
    with torch.inference_mode():
        e32 = np.concatenate([eager(torch.from_numpy(x32[i:i + 256])).numpy()
                              for i in range(0, n, 256)]).astype(np.float64)
        ed = eager.double()
        t64 = np.concatenate([ed(torch.from_numpy(x32[i:i + 256].astype(np.float64))).numpy()
                              for i in range(0, n, 256)])
        eager.float()
        x1 = torch.from_numpy(x32[:1])
        ms_eager = timed(lambda: eager(x1), 200)
    n32 = np.empty(n)
    t0 = time.perf_counter()
    for j in range(n):
        n32[j] = float(np32.forward(x32[j:j + 1])[0])
    ms_np32 = (time.perf_counter() - t0) / n * 1e3
    n32b = np.concatenate([np32.forward(x32[i:i + 256]) for i in range(0, n, 256)]).astype(np.float64)
    n64 = np.concatenate([np64.forward(x32[i:i + 256]) for i in range(0, n, 256)])
    ms_np64 = timed(lambda: np64.forward(x32[:1]), 200)

    def d(u, v):
        e = np.abs(u - v)
        return {'max': float(e.max()), 'p99': float(np.quantile(e, 0.99)),
                'median': float(np.median(e))}

    ref_v, ref_r = _verdicts(t32, th), _ranks(t32, groups)
    res = {
        'book': a.book, 'n_windows': int(n), 'n_timestamps': int(len(np.unique(groups))),
        'n_skipped_nonfinite': int(Z['n_nonfinite']), 'seq_len': int(T),
        'input_dim': int(config['input_dim']), 'hidden_dim': int(config['hidden_dim']),
        'n_heads': int(config.get('n_heads', 4)), 'trade_threshold': th,
        'n_params': int(sum(p.numel() for p in eager.parameters())),
        'torch_version': torch.__version__, 'torch_threads': torch.get_num_threads(),
        'pred_abs_median': float(np.median(np.abs(t32))),
        'min_abs_margin_to_threshold': float(np.min(np.abs(np.abs(t32) - th))),
        'min_abs_pred': float(np.min(np.abs(t32))),
        'diff': {
            'np32_vs_torch32_jit': d(n32, t32),
            'np64_vs_torch32_jit': d(n64, t32),
            'np64_vs_torch64_eager': d(n64, t64),
            'torch32_vs_torch64 (torch own fp32 error)': d(t32, t64),
            'torch32_jit_vs_torch32_eager_batched': d(t32, e32),
            'np32_B1_vs_np32_batched': d(n32, n32b),
        },
        'changes_vs_torch32_jit': {},
        'ms_per_forward_B1': {'torch_jit': ms_jit, 'torch_eager': ms_eager,
                              'numpy_f32': ms_np32, 'numpy_f64': ms_np64},
    }
    for name, p in (('np32', n32), ('np64', n64)):
        res['changes_vs_torch32_jit'][name] = {
            'sign': int(np.sum(np.sign(p) != np.sign(t32))),
            'threshold_verdict': int(np.sum(_verdicts(p, th) != ref_v)),
            'rank_within_timestamp': int(np.sum(_ranks(p, groups) != ref_r)),
        }
    out = a.out_json or os.path.join(work, f'{a.book}_parity.json')
    with open(out, 'w') as f:
        json.dump(res, f, indent=2)
    _print_table(res)
    print(f'[parity] json -> {out}')


def _print_table(r):
    print(f"\n== X7 parity: {r['book']}  n={r['n_windows']} windows / "
          f"{r['n_timestamps']} ts  T={r['seq_len']} F={r['input_dim']} "
          f"H={r['hidden_dim']} heads={r['n_heads']} params={r['n_params']} "
          f"th={r['trade_threshold']}")
    print(f"{'comparison':44s} {'max':>10s} {'p99':>10s} {'median':>10s}")
    for k, v in r['diff'].items():
        print(f"{k:44s} {v['max']:10.3e} {v['p99']:10.3e} {v['median']:10.3e}")
    for k, v in r['changes_vs_torch32_jit'].items():
        print(f"changes {k} vs torch32-jit: sign={v['sign']} "
              f"threshold={v['threshold_verdict']} rank={v['rank_within_timestamp']}")
    print('ms/forward (B=1): ' + ', '.join(
        f'{k}={v:.2f}' for k, v in r['ms_per_forward_B1'].items()))
    print(f"min |pred-th| margin={r['min_abs_margin_to_threshold']:.3e}  "
          f"min |pred|={r['min_abs_pred']:.3e}  median |pred|={r['pred_abs_median']:.3f}")


def _cmd_rss(a):
    """One fresh-process RSS probe; prints a JSON line of checkpoints."""
    import sys
    import time
    cp = {'start': _vmrss_mb()}
    P = _paths(a.artifacts_dir, a.book)
    sys.path.insert(0, _REPO)
    if a.context == 'bot':
        # everything predict_now imports besides torch/model_v2, plus the
        # LightGBM leg's library and the pickled scaler
        import joblib
        import pandas  # noqa: F401
        cp['pandas_joblib'] = _vmrss_mb()
        import lightgbm  # noqa: F401
        cp['lightgbm'] = _vmrss_mb()
        import indicators  # noqa: F401
        import market_data  # noqa: F401
        import prediction_cache  # noqa: F401
        import serving_cache  # noqa: F401
        joblib.load(P['scaler'])
        cp['repo_modules_scaler'] = _vmrss_mb()
    rng = np.random.default_rng(0)
    if a.mode == 'torch':
        import torch
        cp['import_torch'] = _vmrss_mb()
        torch.set_num_threads(int(os.environ.get('TORCH_NUM_THREADS', '2')))
        meta = read_meta(a.npz)   # architecture from the npz: no joblib here
        from model_v2 import RegressionLSTM
        m = RegressionLSTM(input_dim=meta['input_dim'], hidden_dim=meta['hidden_dim'],
                           num_layers=meta['num_layers'], n_heads=meta['n_heads'])
        m.load_state_dict(torch.load(P['model'], map_location='cpu', weights_only=True))
        m.eval()
        x = torch.from_numpy(rng.standard_normal(
            (1, a.seq_len, meta['input_dim'])).astype(np.float32))
        m = torch.jit.trace(m, x, check_trace=False)   # predict_now.load_model does this
        with torch.inference_mode():
            m(x)
            cp['loaded_plus_1_forward'] = _vmrss_mb()
            t0 = time.perf_counter()
            for _ in range(a.reps):
                m(x)
    else:
        net = NumpyLSTM.from_npz(a.npz, np.float32)
        x = rng.standard_normal((1, a.seq_len, net.input_dim)).astype(np.float32)
        net.forward(x)
        cp['loaded_plus_1_forward'] = _vmrss_mb()
        t0 = time.perf_counter()
        for _ in range(a.reps):
            net.forward(x)
    ms = (time.perf_counter() - t0) / max(a.reps, 1) * 1e3
    print(json.dumps({'mode': a.mode, 'context': a.context, 'book': a.book,
                      'rss_mb': cp, 'ms_per_forward': ms,
                      'torch_imported': 'torch' in sys.modules}))


def _cmd_memory(a):
    """Median-of-reps fresh-subprocess RSS for numpy vs torch serving."""
    import subprocess
    import sys
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='',
               TORCH_NUM_THREADS=os.environ.get('TORCH_NUM_THREADS', '2'))
    rows = {}
    for context in ('min', 'bot'):
        for mode in ('numpy', 'torch'):
            runs = []
            for _ in range(a.reps):
                out = subprocess.run(
                    [sys.executable, os.path.abspath(__file__), 'rss', '--mode', mode,
                     '--context', context, '--book', a.book, '--artifacts-dir',
                     a.artifacts_dir, '--npz', a.npz, '--seq-len', str(a.seq_len)],
                    check=True, capture_output=True, text=True, env=env).stdout
                runs.append(json.loads(out.strip().splitlines()[-1]))
            med = {k: float(np.median([r['rss_mb'][k] for r in runs]))
                   for k in runs[0]['rss_mb']}
            rows[f'{context}/{mode}'] = {
                'rss_mb_median': med,
                'final_rss_runs': [r['rss_mb']['loaded_plus_1_forward'] for r in runs],
                'ms_per_forward_median': float(np.median([r['ms_per_forward'] for r in runs])),
                'torch_imported': any(r['torch_imported'] for r in runs)}
    for context in ('min', 'bot'):
        rows[f'{context}/saving_mb'] = (
            rows[f'{context}/torch']['rss_mb_median']['loaded_plus_1_forward']
            - rows[f'{context}/numpy']['rss_mb_median']['loaded_plus_1_forward'])
    print(json.dumps(rows, indent=2))
    if a.out_json:
        with open(a.out_json, 'w') as f:
            json.dump(rows, f, indent=2)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description='X7 numpy-LSTM parity/memory harness')
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('extract', 'parity', 'rss', 'memory'):
        s = sub.add_parser(name)
        s.add_argument('--book', choices=('crypto', 'stock'), default='crypto')
        s.add_argument('--artifacts-dir', default=_REPO)
        if name in ('extract', 'parity'):
            s.add_argument('--n-ts', type=int, default=300)
            s.add_argument('--seed', type=int, default=7)
        if name == 'extract':
            s.add_argument('--out', required=True)
        if name == 'parity':
            s.add_argument('--work-dir', default=None)
            s.add_argument('--out-json', default=None)
        if name in ('rss', 'memory'):
            s.add_argument('--npz', required=True)
            s.add_argument('--seq-len', type=int, required=True)
        if name == 'rss':
            s.add_argument('--mode', choices=('numpy', 'torch'), required=True)
            s.add_argument('--context', choices=('min', 'bot'), default='min')
            s.add_argument('--reps', type=int, default=200)
        if name == 'memory':
            s.add_argument('--reps', type=int, default=3)
            s.add_argument('--out-json', default=None)
    a = ap.parse_args(argv)
    {'extract': _cmd_extract, 'parity': _cmd_parity,
     'rss': _cmd_rss, 'memory': _cmd_memory}[a.cmd](a)


if __name__ == '__main__':
    main()
