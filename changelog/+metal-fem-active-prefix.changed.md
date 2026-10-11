On Metal, `warp.fem` stops the work over fixed-capacity geometry partitions (`ExplicitGeometryPartition` with
`max_cell_count`), environment-first space partitions, space restrictions, integration and BSR assembly at the
part in use (device-side counts), and rebuildable `Nanogrid` refreshes sort only the node candidates of the
active cells; GPU volume rebuilds zero only the node ranges in use. Results are bit for bit the same; the work
no longer scales with the reserved capacity.
