# Refinery model data dictionary

Synthetic continuous LP for qualification. Not an MRPL production model.

| Name | Kind | Meaning | Units |
|---|---|---|---|
| A, B | variables | Purchase/process of two crudes | t |
| COST | objective | Minimize purchase cost | currency |
| CDU | row | Combined crude distillation throughput | t |
| AVAIL_A, AVAIL_B | rows | Crude availability | t |
| PETROL, DIESEL, ATF | rows | Minimum product yields | t |
| SULFUR | row | Blend sulfur mass | t |

Yields are linear. No integer unit-commitment, no nonlinear blending indices.
