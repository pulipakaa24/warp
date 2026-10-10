Build `wp.Bvh` and `wp.Mesh` trees on Metal devices with the LBVH constructor, the default as on CUDA, instead of
falling back to SAH. The tree is the one Warp's CUDA LBVH builder produces from the same bounds (node order, leaf
packing and bounds bit for bit), so closest-point queries break ties the same way on both backends;
`Bvh.rebuild()` on Metal rebuilds an LBVH in place, as on CUDA.
