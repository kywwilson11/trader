"""lstm_numpy_serve (ENGINE X7): numpy replica of RegressionLSTM.forward.

Pure-numpy tests run everywhere (tiny synthetic weights); the torch parity
tests importorskip torch (Jetson/CI). The source-text tests pin that the
module stays research-only: nothing on the serving path imports it until an
owner flips the proposed TRADER_NUMPY_LSTM_SERVE flag.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import lstm_numpy_serve as lns  # noqa: E402


def _synthetic_sd(F=5, H=8, L=2, M=6, seed=0, scale=0.4):
    rng = np.random.default_rng(seed)
    r = lambda *s: (rng.standard_normal(s) * scale).astype(np.float32)  # noqa: E731
    sd = {}
    for l in range(L):
        sd[f'lstm.weight_ih_l{l}'] = r(4 * H, F if l == 0 else H)
        sd[f'lstm.weight_hh_l{l}'] = r(4 * H, H)
        sd[f'lstm.bias_ih_l{l}'] = r(4 * H)
        sd[f'lstm.bias_hh_l{l}'] = r(4 * H)
    sd.update({'attn.in_proj_weight': r(3 * H, H), 'attn.in_proj_bias': r(3 * H),
               'attn.out_proj.weight': r(H, H), 'attn.out_proj.bias': r(H),
               'norm.weight': 1 + r(H), 'norm.bias': r(H),
               'fc.0.weight': r(M, H), 'fc.0.bias': r(M),
               'fc.3.weight': r(1, M), 'fc.3.bias': r(1)})
    return sd


def _naive_forward(sd, x, n_heads, eps=1e-5):
    """Independent float64 per-sample loop reference (textbook formulas)."""
    sd = {k: v.astype(np.float64) for k, v in sd.items()}
    sig = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731
    outs = []
    for xb in x.astype(np.float64):
        seq, l = xb, 0
        while f'lstm.weight_ih_l{l}' in sd:
            Wi, Wh = sd[f'lstm.weight_ih_l{l}'], sd[f'lstm.weight_hh_l{l}']
            bi, bh = sd[f'lstm.bias_ih_l{l}'], sd[f'lstm.bias_hh_l{l}']
            H = Wh.shape[1]
            h, c, hs = np.zeros(H), np.zeros(H), []
            for xt in seq:
                g = Wi @ xt + bi + Wh @ h + bh
                i, f, gg, o = sig(g[:H]), sig(g[H:2*H]), np.tanh(g[2*H:3*H]), sig(g[3*H:])
                c = f * c + i * gg
                h = o * np.tanh(c)
                hs.append(h)
            seq, l = np.array(hs), l + 1
        X = seq
        H = X.shape[1]
        hd = H // n_heads
        qkv = X @ sd['attn.in_proj_weight'].T + sd['attn.in_proj_bias']
        heads = []
        for j in range(n_heads):
            s = slice(j * hd, (j + 1) * hd)
            q, k, v = qkv[:, s], qkv[:, H + s.start:H + s.stop], qkv[:, 2*H + s.start:2*H + s.stop]
            a = q @ k.T / np.sqrt(hd)
            a = np.exp(a - a.max(1, keepdims=True))
            heads.append((a / a.sum(1, keepdims=True)) @ v)
        A = np.concatenate(heads, 1) @ sd['attn.out_proj.weight'].T + sd['attn.out_proj.bias']
        Z = A + X
        Z = (Z - Z.mean(1, keepdims=True)) / np.sqrt(Z.var(1, keepdims=True) + eps)
        p = (Z * sd['norm.weight'] + sd['norm.bias']).mean(0)
        h1 = np.maximum(sd['fc.0.weight'] @ p + sd['fc.0.bias'], 0)
        outs.append((sd['fc.3.weight'] @ h1 + sd['fc.3.bias'])[0])
    return np.array(outs)


@pytest.fixture
def npz(tmp_path):
    path = tmp_path / 'model_v2.npz'
    lns.export_weights(_synthetic_sd(), path, n_heads=2)
    return path


# ---------------------------------------------------------------- pure numpy

def test_forward_matches_independent_reference_float64(npz):
    x = np.random.default_rng(1).standard_normal((9, 11, 5))
    net = lns.NumpyLSTM.from_npz(npz, np.float64)
    ref = _naive_forward(_synthetic_sd(), x, n_heads=2)
    np.testing.assert_allclose(net.forward(x), ref, rtol=0, atol=1e-12)


def test_float32_path_close_to_float64(npz):
    x = np.random.default_rng(2).standard_normal((16, 7, 5)).astype(np.float32)
    y32 = lns.NumpyLSTM.from_npz(npz).forward(x)
    y64 = lns.NumpyLSTM.from_npz(npz, np.float64).forward(x)
    assert y32.dtype == np.float32 and y64.dtype == np.float64
    assert np.max(np.abs(y32 - y64)) < 1e-5


def test_shape_dtype_contract_and_batch_independence(npz):
    net = lns.NumpyLSTM.from_npz(npz)
    x = np.random.default_rng(3).standard_normal((4, 6, 5))
    y = net(x)
    assert y.shape == (4,) and y.dtype == np.float32
    rows = np.array([net.forward(x[i:i + 1])[0] for i in range(4)])
    np.testing.assert_allclose(rows, y, rtol=0, atol=1e-6)
    with pytest.raises(ValueError):
        net.forward(x[0])                      # 2-D input
    with pytest.raises(ValueError):
        net.forward(np.zeros((1, 6, 4)))      # wrong feature count
    with pytest.raises(ValueError):
        lns.NumpyLSTM.from_npz(npz, np.float16)


def test_npz_round_trip_and_meta(npz):
    sd = _synthetic_sd()
    with np.load(npz, allow_pickle=False) as z:
        keys = set(z.files) - {'__meta__'}
        assert keys == set(sd)
        for k in sd:
            assert z[k].dtype == np.float32
            np.testing.assert_array_equal(z[k], sd[k])
    meta = lns.read_meta(npz)
    assert meta == {**meta, 'arch': 'RegressionLSTM', 'format_version': 1,
                    'input_dim': 5, 'hidden_dim': 8, 'num_layers': 2,
                    'fc_hidden': 6, 'n_heads': 2, 'ln_eps': 1e-5,
                    'weight_dtype': 'float32'}
    assert not list(npz.parent.glob('*.tmp.*'))


def test_export_fail_closed_on_unknown_architecture(tmp_path):
    sd = _synthetic_sd()
    with pytest.raises(ValueError):
        lns.export_weights(sd, tmp_path / 'a.npz')            # n_heads unknown
    with pytest.raises(ValueError):
        lns.export_weights(sd, tmp_path / 'b.npz', n_heads=3)  # 8 % 3 != 0
    bad = dict(sd, **{'lstm.weight_ih_l0_reverse': sd['lstm.weight_ih_l0']})
    with pytest.raises(ValueError, match='extra'):
        lns.export_weights(bad, tmp_path / 'c.npz', n_heads=2)
    bad = {k: v for k, v in sd.items() if k != 'norm.bias'}
    with pytest.raises(ValueError, match='missing'):
        lns.export_weights(bad, tmp_path / 'd.npz', n_heads=2)
    bad = dict(sd, **{'fc.3.weight': np.zeros((2, 6), np.float32)})
    with pytest.raises(ValueError):
        lns.export_weights(bad, tmp_path / 'e.npz', n_heads=2)
    assert not list(tmp_path.iterdir())


def test_is_fresh_tracks_source_hash(tmp_path):
    src = tmp_path / 'model_v2.pth'
    src.write_bytes(b'weights-v1')
    out = tmp_path / 'model_v2.npz'
    lns.export_weights(_synthetic_sd(), out, n_heads=2, source_pth=src)
    assert lns.is_fresh(out, src)
    src.write_bytes(b'weights-v2')
    assert not lns.is_fresh(out, src)
    assert not lns.is_fresh(tmp_path / 'missing.npz', src)


def test_module_import_does_not_import_torch():
    code = ('import sys; sys.path.insert(0, %r); import lstm_numpy_serve; '
            'print("torch" in sys.modules)' % str(ROOT))
    out = subprocess.run([sys.executable, '-c', code], capture_output=True,
                         text=True, check=True).stdout.strip()
    assert out == 'False'


def test_serving_path_does_not_use_module_yet():
    """Research-only until the owner flip: no production module imports it."""
    offenders = []
    for p in list(ROOT.glob('*.py')) + list((ROOT / 'scripts').glob('*.py')):
        if p.name == 'lstm_numpy_serve.py':
            continue
        if 'lstm_numpy_serve' in p.read_text(errors='replace'):
            offenders.append(p.name)
    assert offenders == []
    assert 'lstm_numpy_serve' not in (ROOT / 'predict_now.py').read_text()


# ------------------------------------------------------ torch parity (Jetson)

def _torch_model(F, H, L, heads, seed):
    torch = pytest.importorskip('torch')
    from model_v2 import RegressionLSTM
    torch.manual_seed(seed)
    return RegressionLSTM(input_dim=F, hidden_dim=H, num_layers=L,
                          dropout=0.3, n_heads=heads).eval()


@pytest.mark.parametrize('F,H,L,heads', [(7, 16, 2, 4), (5, 12, 1, 1), (9, 24, 3, 8)])
def test_parity_vs_torch_float32_and_float64(tmp_path, F, H, L, heads):
    torch = pytest.importorskip('torch')
    m = _torch_model(F, H, L, heads, seed=F + H)
    path = tmp_path / 'm.npz'
    meta = lns.export_weights(m, path)
    assert meta['n_heads'] == heads
    x = np.random.default_rng(H).standard_normal((64, 12, F)).astype(np.float32)
    with torch.inference_mode():
        t32 = m(torch.from_numpy(x)).numpy()
        t64 = m.double()(torch.from_numpy(x.astype(np.float64))).numpy()
    n32 = lns.NumpyLSTM.from_npz(path, np.float32).forward(x)
    n64 = lns.NumpyLSTM.from_npz(path, np.float64).forward(x)
    assert np.max(np.abs(n32 - t32)) <= 1e-6
    assert np.max(np.abs(n64 - t64)) <= 1e-10


def test_parity_vs_jit_traced_model_and_saturated_inputs(tmp_path):
    torch = pytest.importorskip('torch')
    m = _torch_model(6, 16, 2, 2, seed=3)
    x = (np.random.default_rng(5).standard_normal((32, 10, 6)) * 60).astype(np.float32)
    tm = torch.jit.trace(m, torch.from_numpy(x[:1]), check_trace=False)
    path = tmp_path / 'jit.npz'
    lns.export_weights(tm.state_dict(), path, n_heads=2)
    with torch.inference_mode():
        ref = tm(torch.from_numpy(x)).numpy()
    with np.errstate(all='raise', under='ignore'):
        got = lns.NumpyLSTM.from_npz(path).forward(x)
    assert np.all(np.isfinite(got))
    assert np.max(np.abs(got - ref)) <= 1e-6


def test_export_from_pth_round_trip(tmp_path):
    torch = pytest.importorskip('torch')
    m = _torch_model(4, 8, 2, 2, seed=11)
    pth = tmp_path / 'model_v2.pth'
    torch.save(m.state_dict(), pth)
    out = tmp_path / 'model_v2.npz'
    meta = lns.export_from_pth(pth, out, n_heads=2)
    assert meta['source_pth'] == 'model_v2.pth'
    assert lns.is_fresh(out, pth)
    x = np.random.default_rng(0).standard_normal((3, 5, 4)).astype(np.float32)
    with torch.inference_mode():
        ref = m(torch.from_numpy(x)).numpy()
    assert np.max(np.abs(lns.NumpyLSTM.from_npz(out).forward(x) - ref)) <= 1e-6
