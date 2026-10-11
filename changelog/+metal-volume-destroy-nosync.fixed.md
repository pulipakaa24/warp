Destroying a ``wp.Volume`` on Metal no longer waits for the device: the grid buffer is released once no submitted GPU work can read it.
