# Capillary trace-air regression inputs

`capillary-trace-air.jsonl.gz` contains the 656 captured N₂O capillary UV inputs
that exposed the trace-vapor recovery/enthalpy issue. Gzip decompression yields
the original JSONL byte-for-byte (SHA-256
`ddab6c16a14ee57d7ec9aa7b21e51b6ae3ea33465459a951a83e376aaf21234c`).
The runtime paths and failure statuses inside are historical provenance, not
commands to execute or expected failures of the repaired implementation.

Use the repository's
`examples/impinging-n2o-supercooled/cold-pr-148K-config.yaml` with
`tools/validate_wale_pr_properties.py`. Its mechanism and physical settings
match the captured fixture; only the resolved mechanism path differs.
The mechanism SHA-256 is
`b73e7835a30cc3826b9b0573cf7877bcfe9d6480455d9a86718a40da10980916`.

The property test restores the states on CUDA and compares GPU effective
species enthalpies to the host PR reference, including zero-concentration
species. These 656 rows are correlated states/retries from a single test,
not 656 independent physical validation cases.
