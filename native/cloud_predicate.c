/* Project-owned bounded radius predicate. No third-party implementation code. */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#if defined(__FAST_MATH__) || !defined(FEMTO_STRICT_FP)
#error This kernel requires the recorded strict floating point build flags.
#endif

#define MAX_PAIRS 65536
#if DBL_MANT_DIG != 53 || FLT_RADIX != 2
#error Binary64 is required.
#endif
typedef char check_double_width[sizeof(double) == 8 ? 1 : -1];

static int view(PyObject *object, Py_buffer *buffer, int dimensions,
                const char *format, Py_ssize_t itemsize) {
    if (PyObject_GetBuffer(object, buffer, PyBUF_FORMAT | PyBUF_STRIDES) < 0)
        return 0;
    if (buffer->ndim != dimensions || buffer->itemsize != itemsize ||
        !buffer->format || strcmp(buffer->format, format) != 0 ||
        (uintptr_t)buffer->buf % 8 != 0 ||
        !PyBuffer_IsContiguous(buffer, 'C')) {
        PyErr_SetString(PyExc_ValueError, "Unsupported native predicate buffer");
        return 0;
    }
    return 1;
}

static PyObject *classify(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *point_object, *normal_object, *query_object, *id_object, *split_object;
    Py_buffer points = {0}, normals = {0}, queries = {0}, ids = {0}, splits = {0};
    PyObject *result = NULL;
    double radius_squared, radius_guard, height, rounding;
    if (!PyArg_ParseTuple(args, "OOOOOdddd", &point_object, &normal_object,
                          &query_object, &id_object, &split_object,
                          &radius_squared, &radius_guard, &height, &rounding))
        return NULL;
    if (!view(point_object, &points, 2, "d", 8) ||
        !view(normal_object, &normals, 2, "d", 8) ||
        !view(query_object, &queries, 1, "q", 8) ||
        !view(id_object, &ids, 1, "q", 8) ||
        !view(split_object, &splits, 1, "q", 8))
        goto cleanup;
    const Py_ssize_t count = points.shape[0], nq = queries.shape[0], np = ids.shape[0];
    if (points.shape[1] != 3 || normals.shape[1] != 3 || normals.shape[0] != count ||
        nq > MAX_PAIRS || np > MAX_PAIRS || splits.shape[0] != nq + 1 ||
        !isfinite(radius_squared) || !isfinite(radius_guard) ||
        !isfinite(height) || !isfinite(rounding) ||
        radius_squared <= 0.0 || radius_guard < 0.0 || radius_guard >= radius_squared ||
        height <= 0.0 || rounding < 0.0) {
        PyErr_SetString(PyExc_ValueError, "Invalid or oversized native predicate input");
        goto cleanup;
    }
    const int64_t *q = (const int64_t *)queries.buf;
    const int64_t *neighbor = (const int64_t *)ids.buf;
    const int64_t *split = (const int64_t *)splits.buf;
    if (split[0] != 0 || split[nq] != np) {
        PyErr_SetString(PyExc_ValueError, "Invalid radius batch offsets");
        goto cleanup;
    }
    /* Check the complete index buffers, including any early-positive suffix. */
    for (Py_ssize_t i = 0; i < nq; ++i) {
        if (q[i] < 0 || q[i] >= count || split[i] < 0 || split[i + 1] < split[i]) {
            PyErr_SetString(PyExc_ValueError, "Invalid radius query index or offset");
            goto cleanup;
        }
    }
    for (Py_ssize_t i = 0; i < np; ++i) {
        if (neighbor[i] < 0 || neighbor[i] >= count) {
            PyErr_SetString(PyExc_ValueError, "Invalid radius neighbor index");
            goto cleanup;
        }
    }
    result = PyBytes_FromStringAndSize(NULL, nq);
    if (!result) goto cleanup;
    unsigned char *status = (unsigned char *)PyBytes_AS_STRING(result);
    const double *p = (const double *)points.buf, *n = (const double *)normals.buf;
    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t i = 0; i < nq; ++i) {
        const double *origin = p + 3*q[i], *normal = n + 3*q[i];
        unsigned char value = split[i] == split[i + 1] ? 2 : 0;
        for (int64_t j = split[i]; j < split[i + 1]; ++j) {
            const double *point = p + 3*neighbor[j];
            const double dx = point[0] - origin[0];
            const double dy = point[1] - origin[1];
            const double dz = point[2] - origin[2];
            const double squared = (dx*dx + dy*dy) + dz*dz;
            const double ax = dx*normal[0], ay = dy*normal[1], az = dz*normal[2];
            const double h = fabs((ax + ay) + az);
            const double absolute_sum = (fabs(ax) + fabs(ay)) + fabs(az);
            /* Inputs normally are finite. Preserve NaN propagation if edited. */
            const double guard = rounding * (absolute_sum < height ? height : absolute_sum);
            const int inside = squared < radius_squared - radius_guard;
            const int finite = isfinite(h) && isfinite(guard);
            if (inside && finite && h > height + guard) {
                value = 1;
                break;  /* An original-point witness proves this existential predicate. */
            }
            if (fabs(squared - radius_squared) <= radius_guard ||
                (inside && (!finite || fabs(h - height) <= guard)))
                value = 2;
        }
        status[i] = value;
    }
    Py_END_ALLOW_THREADS
cleanup:
    if (points.obj) PyBuffer_Release(&points);
    if (normals.obj) PyBuffer_Release(&normals);
    if (queries.obj) PyBuffer_Release(&queries);
    if (ids.obj) PyBuffer_Release(&ids);
    if (splits.obj) PyBuffer_Release(&splits);
    return result;
}

/* NaNs must propagate like np.minimum/maximum, rather than C fmin/fmax. */
static double minimum(double a, double b) { return isnan(a) || a < b ? a : b; }
static double maximum(double a, double b) { return isnan(a) || a > b ? a : b; }

static int possible_box(const double *p, const double *n, const double *window_low,
                        const double *window_high, const double *low,
                        const double *high, double height, double rounding) {
    double left[3], right[3], guard_terms[3];
    for (int axis = 0; axis < 3; ++axis) {
        const double a = maximum(low[axis], window_low[axis]);
        const double b = minimum(high[axis], window_high[axis]);
        if (a > b) return 0;
        left[axis] = (a - p[axis])*n[axis];
        right[axis] = (b - p[axis])*n[axis];
        guard_terms[axis] = (maximum(fabs(a), fabs(b)) + fabs(p[axis]))*fabs(n[axis]);
    }
    const double lower = (minimum(left[0], right[0]) + minimum(left[1], right[1])) + minimum(left[2], right[2]);
    const double upper = (maximum(left[0], right[0]) + maximum(left[1], right[1])) + maximum(left[2], right[2]);
    const double bound = maximum(fabs(lower), fabs(upper));
    const double guard_sum = (guard_terms[0] + guard_terms[1]) + guard_terms[2];
    const double guard = rounding*maximum(guard_sum, height);
    return !isfinite(bound) || !isfinite(guard) || bound + guard > height;
}

static PyObject *negative_boxes(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *objects[9];
    Py_buffer buffers[9] = {{0}};
    PyObject *result = NULL;
    double radius, height, rounding;
    if (!PyArg_ParseTuple(args, "OOOOOOOOOddd", &objects[0], &objects[1], &objects[2],
        &objects[3], &objects[4], &objects[5], &objects[6], &objects[7], &objects[8],
        &radius, &height, &rounding)) return NULL;
    const int dimensions[9] = {2, 2, 1, 1, 1, 2, 2, 3, 3};
    for (int i = 0; i < 9; ++i) {
        if (!view(objects[i], &buffers[i], dimensions[i], i >= 2 && i <= 4 ? "q" : "d", 8))
            goto cleanup;
    }
    const Py_ssize_t nq = buffers[0].shape[0], nc = buffers[3].shape[0];
    if (nq > 4096 || !nc || buffers[0].shape[1] != 3 || buffers[1].shape[0] != nq ||
        buffers[1].shape[1] != 3 || buffers[2].shape[0] != nq || buffers[4].shape[0] != 27 ||
        buffers[5].shape[0] != nc || buffers[5].shape[1] != 3 ||
        buffers[6].shape[0] != nc || buffers[6].shape[1] != 3 ||
        buffers[7].shape[0] != nc || buffers[7].shape[1] != 8 || buffers[7].shape[2] != 3 ||
        buffers[8].shape[0] != nc || buffers[8].shape[1] != 8 || buffers[8].shape[2] != 3 ||
        !isfinite(radius) || !isfinite(height) || !isfinite(rounding) ||
        radius <= 0 || height <= 0 || rounding < 0) {
        PyErr_SetString(PyExc_ValueError, "Invalid bounded envelope buffers");
        goto cleanup;
    }
    const double *points = (const double *)buffers[0].buf, *normals = (const double *)buffers[1].buf;
    const int64_t *keys = (const int64_t *)buffers[2].buf, *unique = (const int64_t *)buffers[3].buf;
    const int64_t *offsets = (const int64_t *)buffers[4].buf;
    const double *low = (const double *)buffers[5].buf, *high = (const double *)buffers[6].buf;
    const double *sub_low = (const double *)buffers[7].buf, *sub_high = (const double *)buffers[8].buf;
    for (int i = 0; i < 27; ++i) {
        if (offsets[i] < -(INT64_C(1) << 61) || offsets[i] > (INT64_C(1) << 61)) {
            PyErr_SetString(PyExc_ValueError, "Envelope offset exceeds key budget");
            goto cleanup;
        }
    }
    for (Py_ssize_t i = 0; i < nq; ++i) {
        if (keys[i] < -(INT64_C(1) << 61) || keys[i] > (INT64_C(1) << 61)) {
            PyErr_SetString(PyExc_ValueError, "Envelope key exceeds key budget");
            goto cleanup;
        }
    }
    result = PyBytes_FromStringAndSize(NULL, nq);
    if (!result) goto cleanup;
    unsigned char *negative = (unsigned char *)PyBytes_AS_STRING(result);
    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t i = 0; i < nq; ++i) {
        const double *p = points + 3*i, *n = normals + 3*i;
        double window_low[3], window_high[3];
        for (int axis = 0; axis < 3; ++axis) {
            window_low[axis] = nextafter(p[axis] - radius, -INFINITY);
            window_high[axis] = nextafter(p[axis] + radius, INFINITY);
        }
        negative[i] = 1;
        for (int j = 0; j < 27; ++j) {
            const int64_t target = keys[i] + offsets[j];
            Py_ssize_t lo = 0, hi = nc;
            while (lo < hi) {
                const Py_ssize_t mid = lo + (hi - lo)/2;
                if (unique[mid] < target) lo = mid + 1;
                else hi = mid;
            }
            if (lo == nc || unique[lo] != target) continue;
            if (!possible_box(p, n, window_low, window_high, low + 3*lo, high + 3*lo, height, rounding))
                continue;
            int possible = 0;
            for (int sub = 0; sub < 8; ++sub) {
                if (possible_box(p, n, window_low, window_high, sub_low + 24*lo + 3*sub,
                                 sub_high + 24*lo + 3*sub, height, rounding)) {
                    possible = 1;
                    break;
                }
            }
            if (possible) {
                negative[i] = 0;
                break;
            }
        }
    }
    Py_END_ALLOW_THREADS
cleanup:
    for (int i = 0; i < 9; ++i) if (buffers[i].obj) PyBuffer_Release(&buffers[i]);
    return result;
}

static PyMethodDef methods[] = {
    {"classify", classify, METH_VARARGS, "Return 0=negative, 1=positive, 2=legacy fallback."},
    {"negative_boxes", negative_boxes, METH_VARARGS, "Return the unchanged eight-subbox negative certificates."},
    {NULL, NULL, 0, NULL}
};
static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "_cloud_predicate", NULL, -1, methods};
PyMODINIT_FUNC PyInit__cloud_predicate(void) {
    PyObject *result = PyModule_Create(&module);
    if (result && (PyModule_AddIntConstant(result, "ABI_VERSION", 2) < 0 ||
                   PyModule_AddStringConstant(result, "SOURCE_SHA256", FEMTO_SOURCE_SHA256) < 0)) {
        Py_DECREF(result);
        return NULL;
    }
    return result;
}
