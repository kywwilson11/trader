/*
 * C implementations of hot-path technical indicators for ARM64 (Jetson Orin Nano).
 *
 * These replace Numba JIT functions with native C for:
 *   - Zero JIT compilation overhead (Numba first-call penalty ~2-5s)
 *   - Better ARM64 NEON auto-vectorization
 *   - Lower memory footprint (no LLVM JIT cache)
 *
 * Python fallbacks remain in indicators.py if this module fails to import.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <float.h>

/* ── Helper: check 1D float64 contiguous array ─────────────────────────── */

static int check_array(PyArrayObject *arr, const char *name) {
    if (!PyArray_IS_C_CONTIGUOUS(arr)) {
        PyErr_Format(PyExc_ValueError, "%s must be C-contiguous", name);
        return -1;
    }
    if (PyArray_NDIM(arr) != 1) {
        PyErr_Format(PyExc_ValueError, "%s must be 1-dimensional", name);
        return -1;
    }
    if (PyArray_TYPE(arr) != NPY_DOUBLE) {
        PyErr_Format(PyExc_TypeError, "%s must be float64", name);
        return -1;
    }
    return 0;
}

/* ── EWM with alpha (matches pandas ewm(alpha=..., min_periods=...)) ──── */

static void ewm_alpha(const double *arr, double *out, npy_intp n,
                       double alpha, int min_periods) {
    double s = NAN;
    int count = 0;
    for (npy_intp i = 0; i < n; i++) {
        double v = arr[i];
        if (isnan(v)) {
            out[i] = NAN;
            continue;
        }
        count++;
        if (isnan(s))
            s = v;
        else
            s = alpha * v + (1.0 - alpha) * s;
        out[i] = (count >= min_periods) ? s : NAN;
    }
}

/* ── EWM with span (adjust=False) ────────────────────────────────────── */

static void ewm_span(const double *arr, double *out, npy_intp n, double span) {
    double alpha = 2.0 / (span + 1.0);
    out[0] = arr[0];
    for (npy_intp i = 1; i < n; i++)
        out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1];
}

/* ── Rolling mean ────────────────────────────────────────────────────── */

static void rolling_mean(const double *arr, double *out, npy_intp n, int window) {
    double s = 0.0;
    int valid_run = 0;
    for (npy_intp i = 0; i < n; i++) {
        if (isnan(arr[i])) {
            out[i] = NAN;
            s = 0.0;
            valid_run = 0;
            continue;
        }
        s += arr[i];
        valid_run++;
        if (valid_run > window)
            s -= arr[i - window];
        out[i] = (valid_run >= window) ? s / window : NAN;
    }
}

/* ── Rolling std (ddof=1, matches pandas) ──────────────────────────── */

static void rolling_std(const double *arr, double *out, npy_intp n, int window) {
    for (npy_intp i = 0; i < window - 1; i++)
        out[i] = NAN;
    for (npy_intp i = window - 1; i < n; i++) {
        double s = 0.0, s2 = 0.0;
        for (int j = i - window + 1; j <= i; j++) {
            s += arr[j];
            s2 += arr[j] * arr[j];
        }
        double mean = s / window;
        double var = s2 / window - mean * mean;
        out[i] = (window > 1) ? sqrt(var * window / (window - 1)) : 0.0;
    }
}

/* ── Rolling min/max ─────────────────────────────────────────────────── */

static void rolling_min(const double *arr, double *out, npy_intp n, int window) {
    for (npy_intp i = 0; i < window - 1; i++)
        out[i] = NAN;
    for (npy_intp i = window - 1; i < n; i++) {
        double m = arr[i];
        for (int j = i - window + 1; j < i; j++)
            if (arr[j] < m) m = arr[j];
        out[i] = m;
    }
}

static void rolling_max(const double *arr, double *out, npy_intp n, int window) {
    for (npy_intp i = 0; i < window - 1; i++)
        out[i] = NAN;
    for (npy_intp i = window - 1; i < n; i++) {
        double m = arr[i];
        for (int j = i - window + 1; j < i; j++)
            if (arr[j] > m) m = arr[j];
        out[i] = m;
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * Python-callable functions
 * ═══════════════════════════════════════════════════════════════════════ */

/* ── RSI ────────────────────────────────────────────────────────────── */

static PyObject* py_rsi(PyObject *self, PyObject *args) {
    PyArrayObject *close_arr;
    int length;
    if (!PyArg_ParseTuple(args, "O!i", &PyArray_Type, &close_arr, &length))
        return NULL;
    if (check_array(close_arr, "close")) return NULL;

    npy_intp n = PyArray_SIZE(close_arr);
    const double *close = (double *)PyArray_DATA(close_arr);

    npy_intp dims[1] = {n};
    PyArrayObject *result = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (!result) return NULL;
    double *rsi = (double *)PyArray_DATA(result);

    /* Compute gain/loss */
    double *gain = (double *)malloc(n * sizeof(double));
    double *loss = (double *)malloc(n * sizeof(double));
    double *avg_g = (double *)malloc(n * sizeof(double));
    double *avg_l = (double *)malloc(n * sizeof(double));
    if (!gain || !loss || !avg_g || !avg_l) {
        free(gain); free(loss); free(avg_g); free(avg_l);
        Py_DECREF(result);
        return PyErr_NoMemory();
    }

    gain[0] = 0.0; loss[0] = 0.0;
    for (npy_intp i = 1; i < n; i++) {
        double d = close[i] - close[i - 1];
        gain[i] = d > 0 ? d : 0.0;
        loss[i] = d < 0 ? -d : 0.0;
    }

    double alpha = 1.0 / length;
    ewm_alpha(gain, avg_g, n, alpha, length);
    ewm_alpha(loss, avg_l, n, alpha, length);

    rsi[0] = NAN;
    for (npy_intp i = 1; i < n; i++) {
        if (isnan(avg_g[i]) || isnan(avg_l[i]) || avg_l[i] == 0.0)
            rsi[i] = NAN;
        else {
            double rs = avg_g[i] / avg_l[i];
            rsi[i] = 100.0 - (100.0 / (1.0 + rs));
        }
    }

    free(gain); free(loss); free(avg_g); free(avg_l);
    return (PyObject *)result;
}

/* ── MACD ──────────────────────────────────────────────────────────── */

static PyObject* py_macd(PyObject *self, PyObject *args) {
    PyArrayObject *close_arr;
    int fast, slow, signal;
    if (!PyArg_ParseTuple(args, "O!iii", &PyArray_Type, &close_arr,
                          &fast, &slow, &signal))
        return NULL;
    if (check_array(close_arr, "close")) return NULL;

    npy_intp n = PyArray_SIZE(close_arr);
    const double *close = (double *)PyArray_DATA(close_arr);

    npy_intp dims[1] = {n};
    PyArrayObject *macd_out = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    PyArrayObject *hist_out = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    PyArrayObject *sig_out  = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (!macd_out || !hist_out || !sig_out) {
        Py_XDECREF(macd_out); Py_XDECREF(hist_out); Py_XDECREF(sig_out);
        return PyErr_NoMemory();
    }

    double *ema_f = (double *)malloc(n * sizeof(double));
    double *ema_s = (double *)malloc(n * sizeof(double));
    if (!ema_f || !ema_s) {
        free(ema_f); free(ema_s);
        Py_DECREF(macd_out); Py_DECREF(hist_out); Py_DECREF(sig_out);
        return PyErr_NoMemory();
    }

    double *macd = (double *)PyArray_DATA(macd_out);
    double *hist = (double *)PyArray_DATA(hist_out);
    double *sig  = (double *)PyArray_DATA(sig_out);

    ewm_span(close, ema_f, n, fast);
    ewm_span(close, ema_s, n, slow);
    for (npy_intp i = 0; i < n; i++)
        macd[i] = ema_f[i] - ema_s[i];
    ewm_span(macd, sig, n, signal);
    for (npy_intp i = 0; i < n; i++)
        hist[i] = macd[i] - sig[i];

    free(ema_f); free(ema_s);
    return Py_BuildValue("(OOO)", macd_out, hist_out, sig_out);
}

/* ── ATR ───────────────────────────────────────────────────────────── */

static PyObject* py_atr(PyObject *self, PyObject *args) {
    PyArrayObject *high_arr, *low_arr, *close_arr;
    int length;
    if (!PyArg_ParseTuple(args, "O!O!O!i",
                          &PyArray_Type, &high_arr,
                          &PyArray_Type, &low_arr,
                          &PyArray_Type, &close_arr, &length))
        return NULL;
    if (check_array(high_arr, "high") || check_array(low_arr, "low") ||
        check_array(close_arr, "close"))
        return NULL;

    npy_intp n = PyArray_SIZE(high_arr);
    const double *high = (double *)PyArray_DATA(high_arr);
    const double *low  = (double *)PyArray_DATA(low_arr);
    const double *close = (double *)PyArray_DATA(close_arr);

    double *tr = (double *)malloc(n * sizeof(double));
    if (!tr) return PyErr_NoMemory();

    tr[0] = high[0] - low[0];
    for (npy_intp i = 1; i < n; i++) {
        double hl = high[i] - low[i];
        double hc = fabs(high[i] - close[i - 1]);
        double lc = fabs(low[i] - close[i - 1]);
        tr[i] = hl;
        if (hc > tr[i]) tr[i] = hc;
        if (lc > tr[i]) tr[i] = lc;
    }

    npy_intp dims[1] = {n};
    PyArrayObject *result = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (!result) { free(tr); return PyErr_NoMemory(); }

    rolling_mean(tr, (double *)PyArray_DATA(result), n, length);
    free(tr);
    return (PyObject *)result;
}

/* ── Bollinger Bands ───────────────────────────────────────────────── */

static PyObject* py_bbands(PyObject *self, PyObject *args) {
    PyArrayObject *close_arr;
    int length;
    double num_std;
    if (!PyArg_ParseTuple(args, "O!id", &PyArray_Type, &close_arr,
                          &length, &num_std))
        return NULL;
    if (check_array(close_arr, "close")) return NULL;

    npy_intp n = PyArray_SIZE(close_arr);
    const double *close = (double *)PyArray_DATA(close_arr);

    npy_intp dims[1] = {n};
    PyArrayObject *lo_arr = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    PyArrayObject *mid_arr = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    PyArrayObject *up_arr = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    PyArrayObject *bw_arr = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    PyArrayObject *pb_arr = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (!lo_arr || !mid_arr || !up_arr || !bw_arr || !pb_arr) {
        Py_XDECREF(lo_arr); Py_XDECREF(mid_arr); Py_XDECREF(up_arr);
        Py_XDECREF(bw_arr); Py_XDECREF(pb_arr);
        return PyErr_NoMemory();
    }

    double *lo = (double *)PyArray_DATA(lo_arr);
    double *mid = (double *)PyArray_DATA(mid_arr);
    double *up = (double *)PyArray_DATA(up_arr);
    double *bw = (double *)PyArray_DATA(bw_arr);
    double *pb = (double *)PyArray_DATA(pb_arr);

    rolling_mean(close, mid, n, length);
    rolling_std(close, up, n, length);  /* temp: reuse up for std */

    for (npy_intp i = 0; i < n; i++) {
        double std_val = up[i];
        up[i] = mid[i] + num_std * std_val;
        lo[i] = mid[i] - num_std * std_val;
        if (isnan(mid[i]) || mid[i] == 0.0)
            bw[i] = NAN;
        else
            bw[i] = (up[i] - lo[i]) / mid[i];
        double diff = up[i] - lo[i];
        if (isnan(diff) || diff == 0.0)
            pb[i] = NAN;
        else
            pb[i] = (close[i] - lo[i]) / diff;
    }

    return Py_BuildValue("(OOOOO)", lo_arr, mid_arr, up_arr, bw_arr, pb_arr);
}

/* ── Stochastic ────────────────────────────────────────────────────── */

static PyObject* py_stoch(PyObject *self, PyObject *args) {
    PyArrayObject *high_arr, *low_arr, *close_arr;
    int k, d, smooth_k;
    if (!PyArg_ParseTuple(args, "O!O!O!iii",
                          &PyArray_Type, &high_arr,
                          &PyArray_Type, &low_arr,
                          &PyArray_Type, &close_arr,
                          &k, &d, &smooth_k))
        return NULL;
    if (check_array(high_arr, "high") || check_array(low_arr, "low") ||
        check_array(close_arr, "close"))
        return NULL;

    npy_intp n = PyArray_SIZE(close_arr);
    const double *high = (double *)PyArray_DATA(high_arr);
    const double *low  = (double *)PyArray_DATA(low_arr);
    const double *close = (double *)PyArray_DATA(close_arr);

    double *lowest = (double *)malloc(n * sizeof(double));
    double *highest = (double *)malloc(n * sizeof(double));
    double *raw_k_arr = (double *)malloc(n * sizeof(double));
    if (!lowest || !highest || !raw_k_arr) {
        free(lowest); free(highest); free(raw_k_arr);
        return PyErr_NoMemory();
    }

    rolling_min(low, lowest, n, k);
    rolling_max(high, highest, n, k);

    for (npy_intp i = 0; i < n; i++) {
        double diff = highest[i] - lowest[i];
        if (isnan(diff) || diff == 0.0)
            raw_k_arr[i] = NAN;
        else
            raw_k_arr[i] = 100.0 * (close[i] - lowest[i]) / diff;
    }

    npy_intp dims[1] = {n};
    PyArrayObject *sk_arr = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    PyArrayObject *sd_arr = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (!sk_arr || !sd_arr) {
        free(lowest); free(highest); free(raw_k_arr);
        Py_XDECREF(sk_arr); Py_XDECREF(sd_arr);
        return PyErr_NoMemory();
    }

    rolling_mean(raw_k_arr, (double *)PyArray_DATA(sk_arr), n, smooth_k);
    rolling_mean((double *)PyArray_DATA(sk_arr),
                 (double *)PyArray_DATA(sd_arr), n, d);

    free(lowest); free(highest); free(raw_k_arr);
    return Py_BuildValue("(OO)", sk_arr, sd_arr);
}

/* ── OBV ───────────────────────────────────────────────────────────── */

static PyObject* py_obv(PyObject *self, PyObject *args) {
    PyArrayObject *close_arr, *volume_arr;
    if (!PyArg_ParseTuple(args, "O!O!",
                          &PyArray_Type, &close_arr,
                          &PyArray_Type, &volume_arr))
        return NULL;
    if (check_array(close_arr, "close") || check_array(volume_arr, "volume"))
        return NULL;

    npy_intp n = PyArray_SIZE(close_arr);
    const double *close = (double *)PyArray_DATA(close_arr);
    const double *volume = (double *)PyArray_DATA(volume_arr);

    npy_intp dims[1] = {n};
    PyArrayObject *result = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (!result) return PyErr_NoMemory();
    double *out = (double *)PyArray_DATA(result);

    out[0] = 0.0;
    for (npy_intp i = 1; i < n; i++) {
        double d = close[i] - close[i - 1];
        if (d > 0)      out[i] = out[i - 1] + volume[i];
        else if (d < 0) out[i] = out[i - 1] - volume[i];
        else             out[i] = out[i - 1];
    }
    return (PyObject *)result;
}

/* ── Rolling percentile (O(n*w) — same complexity but no Python overhead) */

static PyObject* py_rolling_percentile(PyObject *self, PyObject *args) {
    PyArrayObject *arr;
    int window;
    if (!PyArg_ParseTuple(args, "O!i", &PyArray_Type, &arr, &window))
        return NULL;
    if (check_array(arr, "arr")) return NULL;

    npy_intp n = PyArray_SIZE(arr);
    const double *data = (double *)PyArray_DATA(arr);

    npy_intp dims[1] = {n};
    PyArrayObject *result = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (!result) return PyErr_NoMemory();
    double *out = (double *)PyArray_DATA(result);

    for (npy_intp i = 0; i < window - 1; i++)
        out[i] = NAN;

    for (npy_intp i = window - 1; i < n; i++) {
        double val = data[i];
        int count_below = 0, valid = 0;
        for (int j = i - window + 1; j <= i; j++) {
            if (!isnan(data[j])) {
                valid++;
                if (data[j] < val) count_below++;
            }
        }
        out[i] = valid > 0 ? (double)count_below / valid : NAN;
    }
    return (PyObject *)result;
}

/* ── Linear slope ──────────────────────────────────────────────────── */

static PyObject* py_linear_slope(PyObject *self, PyObject *args) {
    PyArrayObject *arr;
    int window;
    if (!PyArg_ParseTuple(args, "O!i", &PyArray_Type, &arr, &window))
        return NULL;
    if (check_array(arr, "arr")) return NULL;

    npy_intp n = PyArray_SIZE(arr);
    const double *data = (double *)PyArray_DATA(arr);

    npy_intp dims[1] = {n};
    PyArrayObject *result = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (!result) return PyErr_NoMemory();
    double *out = (double *)PyArray_DATA(result);

    double x_mean = (window - 1.0) / 2.0;
    double ss_x = 0.0;
    for (int k = 0; k < window; k++)
        ss_x += (k - x_mean) * (k - x_mean);

    for (npy_intp i = 0; i < window - 1; i++)
        out[i] = NAN;

    for (npy_intp i = window - 1; i < n; i++) {
        double y_mean = 0.0;
        for (int j = 0; j < window; j++)
            y_mean += data[i - window + 1 + j];
        y_mean /= window;
        double ss_xy = 0.0;
        for (int j = 0; j < window; j++)
            ss_xy += (j - x_mean) * (data[i - window + 1 + j] - y_mean);
        out[i] = (ss_x != 0.0) ? ss_xy / ss_x : 0.0;
    }
    return (PyObject *)result;
}

/* ── Hurst exponent (R/S analysis) ─────────────────────────────────── */

static PyObject* py_hurst(PyObject *self, PyObject *args) {
    PyArrayObject *arr;
    int window;
    if (!PyArg_ParseTuple(args, "O!i", &PyArray_Type, &arr, &window))
        return NULL;
    if (check_array(arr, "arr")) return NULL;

    npy_intp n = PyArray_SIZE(arr);
    const double *data = (double *)PyArray_DATA(arr);

    npy_intp dims[1] = {n};
    PyArrayObject *result = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (!result) return PyErr_NoMemory();
    double *out = (double *)PyArray_DATA(result);

    /* Pre-allocate temp buffer for cumulative deviation */
    double *cum_dev = (double *)malloc(window * sizeof(double));
    if (!cum_dev) { Py_DECREF(result); return PyErr_NoMemory(); }

    double log_window = log((double)window);

    for (npy_intp i = 0; i < window - 1; i++)
        out[i] = NAN;

    for (npy_intp i = window - 1; i < n; i++) {
        const double *seg = data + (i - window + 1);

        /* Mean */
        double mean_val = 0.0;
        for (int j = 0; j < window; j++)
            mean_val += seg[j];
        mean_val /= window;

        /* Cumulative deviation */
        cum_dev[0] = seg[0] - mean_val;
        double r_max = cum_dev[0], r_min = cum_dev[0];
        for (int j = 1; j < window; j++) {
            cum_dev[j] = cum_dev[j - 1] + (seg[j] - mean_val);
            if (cum_dev[j] > r_max) r_max = cum_dev[j];
            if (cum_dev[j] < r_min) r_min = cum_dev[j];
        }
        double r = r_max - r_min;

        /* Std dev */
        double var = 0.0;
        for (int j = 0; j < window; j++) {
            double d = seg[j] - mean_val;
            var += d * d;
        }
        double s = sqrt(var / window);

        if (s > 1e-10 && r > 0.0)
            out[i] = log(r / s) / log_window;
        else
            out[i] = 0.5;
    }

    free(cum_dev);
    return (PyObject *)result;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Module definition
 * ═══════════════════════════════════════════════════════════════════════ */

static PyMethodDef IndicatorMethods[] = {
    {"rsi",       py_rsi,       METH_VARARGS, "RSI(close, length) -> array"},
    {"macd",      py_macd,      METH_VARARGS, "MACD(close, fast, slow, signal) -> (macd, hist, signal)"},
    {"atr",       py_atr,       METH_VARARGS, "ATR(high, low, close, length) -> array"},
    {"bbands",    py_bbands,    METH_VARARGS, "BBands(close, length, num_std) -> (lo, mid, up, bw, pb)"},
    {"stoch",     py_stoch,     METH_VARARGS, "Stoch(high, low, close, k, d, smooth_k) -> (sk, sd)"},
    {"obv",       py_obv,       METH_VARARGS, "OBV(close, volume) -> array"},
    {"rolling_percentile", py_rolling_percentile, METH_VARARGS, "RollingPercentile(arr, window) -> array"},
    {"linear_slope", py_linear_slope, METH_VARARGS, "LinearSlope(arr, window) -> array"},
    {"hurst",     py_hurst,     METH_VARARGS, "Hurst(arr, window) -> array"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef indicatorsmodule = {
    PyModuleDef_HEAD_INIT,
    "indicators_c",
    "Native C implementations of technical indicators for ARM64",
    -1,
    IndicatorMethods
};

PyMODINIT_FUNC PyInit_indicators_c(void) {
    import_array();
    return PyModule_Create(&indicatorsmodule);
}
