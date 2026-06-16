# Archived LWC/LET Implementation

This directory, now under `fake_quant_learnable/support/`, preserves the previous learnable PTQ implementation for later
reference. It includes the historical LWC, LET, SmoothQuant-initialized
LWC+LET, and tail-protect experiment code.

The active `fake_quant_learnable` package has been reduced to:

- naive W8A8 fake quantization
- SmoothQuant W8A8 fake quantization

Do not import code from this directory in active experiments. Treat these files
as a snapshot of the removed implementation.
