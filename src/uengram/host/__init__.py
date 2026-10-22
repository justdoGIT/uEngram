"""uEngram for running Linux hosts: a typed 1 Hz snapshot of the machine
(PMU counters, /proc, /sys, RAPL, coretemp) windowed into uEngram's memory.

The collector modules (schema, perf, sources, collector, cli record/probe)
use only the standard library; the model modules (features, encoder, node,
cluster) need numpy. See docs/HOST-TELEMETRY.md.
"""
