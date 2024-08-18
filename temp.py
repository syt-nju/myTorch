from numpy import np
def matmul_broadcast(a: np.ndarray, b: np.ndarray):
    assert a.ndim != 0 and b.ndim != 0

    def broadcast(a: np.ndarray, b: np.ndarray):
        if a.ndim < b.ndim:
            b, a = broadcast(b, a)
            return a, b
        for _ in range(a.ndim - b.ndim):
            b = b[np.newaxis, :]
        shape_a = a.shape
        shape_b = b.shape
        for i in range(3, b.ndim + 1): # 这里是对2维以后的维数进行广播
            if shape_a[-i] != shape_b[-i] and shape_a[-i] != 1 and shape_b[
                    -i] != 1:
                return None
            elif shape_a[-i] == 1:
                a = np.repeat(a, shape_b[-i], -i)
            elif shape_b[-i] == 1:
                b = np.repeat(b, shape_a[-i], -i)
        assert a.shape[:-2] == b.shape[:-2]
        return a, b

    if a.ndim == 1 and b.ndim == 1:
        return a @ b
    elif a.ndim == 1:
        a = a[np.newaxis, :]
        a, b = broadcast(a, b)
        c = a @ b
        return c.reshape(c.shape[:-2] + c.shape[-1:])
    elif b.ndim == 1:
        b = b[:, np.newaxis]
        a, b = broadcast(a, b)
        c = a @ b
        return c.reshape(c.shape[:-1])
    else:
        a, b = broadcast(a, b)
        return a @ b