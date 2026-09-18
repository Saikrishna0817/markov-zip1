# Threat model — THREAT-M0-01

Trust boundaries include untrusted model files, archives, manifests, checkpoints, names, paths, environment variables, and benchmark outputs. Risks include overflow-before-allocation, memory/CPU exhaustion, path traversal, symlink escape, decompression bombs, malformed numbers, locale confusion, command injection, unsafe temporary files, secrets in logs, dependency compromise, corrupt evidence, and denial of service.

Controls: checked arithmetic; explicit limits; strict formats; no shell interpolation of untrusted text; canonical paths; reject archive traversal/symlinks; bounded extraction; deterministic manifests; dependency allowlist; no telemetry; no secrets; least privilege; fresh build directories; checksum verification; fail closed; preserve raw evidence. M0 scripts operate only within their resolved project/release roots.
