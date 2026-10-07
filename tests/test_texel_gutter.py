"""Production gutter parity and allocation tests using synthetic arrays."""
import tracemalloc
import numpy as np
import dilate_oracle as LEGACY
from mesh_jig import texel_bake
DILATE = texel_bake.dilate


def test_exact_gutter_and_pixel_parity_with_immutable_inputs():
    rng = np.random.default_rng(1729)
    for shape in [(1,1),(1,9),(9,1),(3,7),(32,32),(0,0)]:
        for density in (0,.1,.9,1):
            colour = rng.random(shape+(3,))
            covered = rng.random(shape)<density
            before = colour.tobytes(),covered.tobytes()
            for steps in (-1,0,1,4,8):
                old,new = LEGACY.dilate(colour,covered,steps),DILATE(colour,covered,steps)
                assert all(a.tobytes()==b.tobytes() for a,b in zip(old,new))
                assert texel_bake.pixels(*old).tobytes()==texel_bake.pixels(*new).tobytes()
                assert (colour.tobytes(),covered.tobytes()) == before


def test_gutter_workspace_budget():
    # Deterministic allocation regression, no machine-speed threshold. Input
    # allocations precede tracing. This fixed fixture allows 72 bytes/texel plus
    # interpreter overhead; full-atlas copies need about 107/texel, the earlier
    # slice-only prototype about 82/texel. Masked ufunc out avoids both temporaries.
    size = 512
    colour = np.zeros((size,size,3),dtype=np.float64)
    covered = np.zeros((size,size),dtype=bool)
    covered[size//4:3*size//4,size//4:3*size//4] = True
    tracemalloc.start()
    try:
        DILATE(colour,covered,8)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 72*size*size + 131072, peak


def test_lower_precision_and_strided_float_inputs_preserve_oracle_bytes():
    rng = np.random.default_rng(1730)
    for dtype in (np.float16,np.float32):
        colour = rng.uniform(-1,1,(12,18,3)).astype(dtype)[::2,::2]
        covered = (rng.random((12,18))<.2)[::2,::2]
        before = colour.tobytes(),covered.tobytes()
        for steps in (0,1,4,8):
            old,new = LEGACY.dilate(colour,covered,steps),DILATE(colour,covered,steps)
            assert all(a.tobytes()==b.tobytes() for a,b in zip(old,new))
            assert texel_bake.pixels(*old).tobytes()==texel_bake.pixels(*new).tobytes()
            assert (colour.tobytes(),covered.tobytes())==before
