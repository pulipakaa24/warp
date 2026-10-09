Compile out the tile reduction of `warp.fem.integrate()`'s dispatched bilinear assembly and of the
`warp.sparse.bsr_mm()` coefficient count when they run with one lane per block. The branch was guarded by a
runtime comparison, so its tile scratch was still reserved; on Metal, 6x6 node blocks (Newton's implicit MPM)
needed 76,944 bytes of threadgroup memory and the launch failed.
