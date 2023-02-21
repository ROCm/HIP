# cuComplex API supported by HIP

## **1. cuComplex Data types**

| **type**     |   **CUDA**                                                    |   **HIP**                                                  |**HIP value** (if differs) |
|-------------:|---------------------------------------------------------------|------------------------------------------------------------|---------------------------|
| float2       |***`cuFloatComplex`***                                         |***`hipFloatComplex`***                                     | struct                    |
| double2      |***`cuDoubleComplex`***                                        |***`hipDoubleComplex`***                                    | struct                    |
| float2       |***`cuComplex`***                                              |***`hipComplex`***                                          | struct                    |

## **2. cuComplex API functions**

|   **CUDA**                                                |   **HIP**                                       |
|-----------------------------------------------------------|-------------------------------------------------|
|`cuCrealf`                                                 |`hipCrealf`                                      |
|`cuCimagf`                                                 |`hipCimagf`                                      |
|`make_cuFloatComplex`                                      |`make_hipFloatComplex`                           |
|`cuConjf`                                                  |`hipConjf`                                       |
|`cuCaddf`                                                  |`hipCaddf`                                       |
|`cuCsubf`                                                  |`hipCsubf`                                       |
|`cuCmulf`                                                  |`hipCmulf`                                       |
|`cuCdivf`                                                  |`hipCdivf`                                       |
|`cuCabsf`                                                  |`hipCabsf`                                       |
|`cuCreal`                                                  |`hipCreal`                                       |
|`cuCimag`                                                  |`hipCimag`                                       |
|`make_cuDoubleComplex`                                     |`make_hipDoubleComplex`                          |
|`cuConj`                                                   |`hipConj`                                        |
|`cuCadd`                                                   |`hipCadd`                                        |
|`cuCsub`                                                   |`hipCsub`                                        |
|`cuCmul`                                                   |`hipCmul`                                        |
|`cuCdiv`                                                   |`hipCdiv`                                        |
|`cuCabs`                                                   |`hipCabs`                                        |
|`make_cuComplex`                                           |`make_hipComplex`                                |
|`cuComplexFloatToDouble`                                   |`hipComplexFloatToDouble`                        |
|`cuComplexDoubleToFloat`                                   |`hipComplexDoubleToFloat`                        |
|`cuCfmaf`                                                  |`hipCfmaf`                                       |
|`cuCfma`                                                   |`hipCfma`                                        |
