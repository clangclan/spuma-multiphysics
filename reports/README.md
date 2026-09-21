# Validation records

Reports in this directory describe the source revision and environment recorded in each report. They are historical evidence, not the current feature list.

The current solver is [ReactiveFoam](../README.ko.md), including [physics selection](reactive-physics-selection-20260921.md). ColdFoam and its dedicated tools were removed after commit `49d476c`; their older measurements remain here for traceability. Source paths and commands in those reports refer to the historical tree and may no longer exist on `main`.

Removing ColdFoam does not transfer its WALE LES or surface-tension models into ReactiveFoam.
