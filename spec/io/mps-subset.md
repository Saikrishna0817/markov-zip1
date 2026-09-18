# MPS subset v1

Supported: ASCII-compatible free format; comments beginning with *; NAME, optional OBJSENSE/OBJNAME, ROWS, COLUMNS, one RHS/RANGES/BOUNDS vector, ENDATA; row types N/E/L/G; INTORG/INTEND; LO/UP/FX/FR/MI/PL/BV/LI/UI. Duplicate matrix coefficients are summed. Unknown extensions are rejected. Limits bound bytes, lines, names, rows, columns, and stored coefficient records.

Additional strictness: names must be printable ASCII without spaces; any content after ENDATA is rejected; objective-row RHS/RANGES values are rejected as dialect-ambiguous; byte accounting is checked before addition to prevent wraparound.
