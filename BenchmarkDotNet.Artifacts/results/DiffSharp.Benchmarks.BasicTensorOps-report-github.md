``` ini

BenchmarkDotNet=v0.12.1, OS=Windows 10.0.17134.1792 (1803/April2018Update/Redstone4)
Intel Xeon CPU E5-1620 0 3.60GHz, 1 CPU, 8 logical and 4 physical cores
.NET Core SDK=5.0.100
  [Host]   : .NET Core 3.1.9 (CoreCLR 4.700.20.47201, CoreFX 4.700.20.47203), X64 RyuJIT DEBUG
  ShortRun : .NET Core 3.1.9 (CoreCLR 4.700.20.47201, CoreFX 4.700.20.47203), X64 RyuJIT

Job=ShortRun  IterationCount=3  LaunchCount=1  
WarmupCount=3  

```
|                           Method |   Categories | tensorSize | dtypeName | deviceName |             Mean |             Error |         StdDev | Ratio | RatioSD | Baseline |
|--------------------------------- |------------- |----------- |---------- |----------- |-----------------:|------------------:|---------------:|------:|--------:|--------- |
|              **fromCpuData_PyTorch** |  **fromCpuData** |         **16** |   **float32** |        **cpu** |    **528,766.53 μs** |      **3,084.493 μs** |     **169.071 μs** | **1.000** |    **0.00** |      **Yes** |
|           fromCpuData_TorchSharp |  fromCpuData |         16 |   float32 |        cpu |    172,461.40 μs |     62,162.891 μs |   3,407.358 μs | 0.326 |    0.01 |       No |
|      fromCpuData_RawTensor_Torch |  fromCpuData |         16 |   float32 |        cpu |    283,193.43 μs |     20,330.706 μs |   1,114.395 μs | 0.536 |    0.00 |       No |
|         fromCpuData_Tensor_Torch |  fromCpuData |         16 |   float32 |        cpu |    270,122.60 μs |     24,305.510 μs |   1,332.267 μs | 0.511 |    0.00 |       No |
|  fromCpuData_RawTensor_Reference |  fromCpuData |         16 |   float32 |        cpu |      4,758.60 μs |        227.301 μs |      12.459 μs | 0.009 |    0.00 |       No |
|     fromCpuData_Tensor_Reference |  fromCpuData |         16 |   float32 |        cpu |      3,718.76 μs |        323.682 μs |      17.742 μs | 0.007 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                    zeros_PyTorch |        zeros |         16 |   float32 |        cpu |  1,645,836.17 μs |      1,652.322 μs |      90.569 μs | 1.000 |    0.00 |      Yes |
|                 zeros_TorchSharp |        zeros |         16 |   float32 |        cpu |    566,064.90 μs |    223,149.633 μs |  12,231.584 μs | 0.344 |    0.01 |       No |
|            zeros_RawTensor_Torch |        zeros |         16 |   float32 |        cpu |    607,117.87 μs |     36,512.034 μs |   2,001.348 μs | 0.369 |    0.00 |       No |
|               zeros_Tensor_Torch |        zeros |         16 |   float32 |        cpu |    625,278.17 μs |    239,994.113 μs |  13,154.887 μs | 0.380 |    0.01 |       No |
|        zeros_RawTensor_Reference |        zeros |         16 |   float32 |        cpu |     12,119.58 μs |      7,180.167 μs |     393.569 μs | 0.007 |    0.00 |       No |
|           zeros_Tensor_Reference |        zeros |         16 |   float32 |        cpu |     12,644.96 μs |      2,742.276 μs |     150.313 μs | 0.008 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                     ones_PyTorch |         ones |         16 |   float32 |        cpu |  1,669,833.47 μs |      2,529.757 μs |     138.664 μs | 1.000 |    0.00 |      Yes |
|                  ones_TorchSharp |         ones |         16 |   float32 |        cpu |    568,870.40 μs |    269,329.897 μs |  14,762.880 μs | 0.341 |    0.01 |       No |
|             ones_RawTensor_Torch |         ones |         16 |   float32 |        cpu |    625,976.93 μs |     58,354.222 μs |   3,198.592 μs | 0.375 |    0.00 |       No |
|                ones_Tensor_Torch |         ones |         16 |   float32 |        cpu |    633,348.77 μs |    248,980.863 μs |  13,647.481 μs | 0.379 |    0.01 |       No |
|         ones_RawTensor_Reference |         ones |         16 |   float32 |        cpu |     12,783.91 μs |      1,096.407 μs |      60.098 μs | 0.008 |    0.00 |       No |
|            ones_Tensor_Reference |         ones |         16 |   float32 |        cpu |     14,479.27 μs |      2,101.341 μs |     115.182 μs | 0.009 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                     rand_PyTorch |         rand |         16 |   float32 |        cpu |  1,981,744.80 μs |      9,183.899 μs |     503.400 μs |  1.00 |    0.00 |      Yes |
|                  rand_TorchSharp |         rand |         16 |   float32 |        cpu |    735,279.23 μs |     46,264.108 μs |   2,535.892 μs |  0.37 |    0.00 |       No |
|             rand_RawTensor_Torch |         rand |         16 |   float32 |        cpu |    734,405.70 μs |    147,900.898 μs |   8,106.947 μs |  0.37 |    0.00 |       No |
|                rand_Tensor_Torch |         rand |         16 |   float32 |        cpu |    739,931.73 μs |     55,914.789 μs |   3,064.878 μs |  0.37 |    0.00 |       No |
|         rand_RawTensor_Reference |         rand |         16 |   float32 |        cpu |     45,266.47 μs |      7,291.756 μs |     399.686 μs |  0.02 |    0.00 |       No |
|            rand_Tensor_Reference |         rand |         16 |   float32 |        cpu |     47,331.28 μs |      4,925.770 μs |     269.998 μs |  0.02 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                 addition_PyTorch |     addition |         16 |   float32 |        cpu |    742,553.60 μs |      5,511.582 μs |     302.108 μs |  1.00 |    0.00 |      Yes |
|              addition_TorchSharp |     addition |         16 |   float32 |        cpu |    553,594.40 μs |     50,133.006 μs |   2,747.959 μs |  0.75 |    0.00 |       No |
|         addition_RawTensor_Torch |     addition |         16 |   float32 |        cpu |    600,035.93 μs |    141,727.998 μs |   7,768.590 μs |  0.81 |    0.01 |       No |
|            addition_Tensor_Torch |     addition |         16 |   float32 |        cpu |  1,063,242.97 μs |    303,698.700 μs |  16,646.751 μs |  1.43 |    0.02 |       No |
|     addition_RawTensor_Reference |     addition |         16 |   float32 |        cpu |     13,933.52 μs |      2,650.274 μs |     145.270 μs |  0.02 |    0.00 |       No |
|        addition_Tensor_Reference |     addition |         16 |   float32 |        cpu |    169,125.67 μs |     30,300.934 μs |   1,660.896 μs |  0.23 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                addScalar_PyTorch |    addScalar |         16 |   float32 |        cpu |  1,938,654.80 μs |      6,392.896 μs |     350.416 μs | 1.000 |    0.00 |      Yes |
|             addScalar_TorchSharp |    addScalar |         16 |   float32 |        cpu |  1,538,171.03 μs |    560,645.124 μs |  30,730.851 μs | 0.793 |    0.02 |       No |
|        addScalar_RawTensor_Torch |    addScalar |         16 |   float32 |        cpu |  1,759,485.30 μs |    165,935.506 μs |   9,095.485 μs | 0.908 |    0.00 |       No |
|           addScalar_Tensor_Torch |    addScalar |         16 |   float32 |        cpu |  2,578,374.60 μs |    323,379.674 μs |  17,725.531 μs | 1.330 |    0.01 |       No |
|    addScalar_RawTensor_Reference |    addScalar |         16 |   float32 |        cpu |      8,175.50 μs |      2,030.730 μs |     111.311 μs | 0.004 |    0.00 |       No |
|       addScalar_Tensor_Reference |    addScalar |         16 |   float32 |        cpu |    159,980.90 μs |     39,756.913 μs |   2,179.211 μs | 0.083 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|             addWithAlpha_PyTorch | addWithAlpha |         16 |   float32 |        cpu |    530,747.83 μs |      5,087.274 μs |     278.851 μs |  1.00 |    0.00 |      Yes |
|          addWithAlpha_TorchSharp | addWithAlpha |         16 |   float32 |        cpu |    475,018.67 μs |    125,692.469 μs |   6,889.628 μs |  0.89 |    0.01 |       No |
|     addWithAlpha_RawTensor_Torch | addWithAlpha |         16 |   float32 |        cpu |  1,323,792.27 μs |    320,908.399 μs |  17,590.072 μs |  2.49 |    0.03 |       No |
|        addWithAlpha_Tensor_Torch | addWithAlpha |         16 |   float32 |        cpu |  2,171,304.50 μs |     90,527.099 μs |   4,962.096 μs |  4.09 |    0.01 |       No |
| addWithAlpha_RawTensor_Reference | addWithAlpha |         16 |   float32 |        cpu |     21,349.36 μs |      2,532.073 μs |     138.791 μs |  0.04 |    0.00 |       No |
|    addWithAlpha_Tensor_Reference | addWithAlpha |         16 |   float32 |        cpu |    276,004.80 μs |      5,242.844 μs |     287.378 μs |  0.52 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|               addInPlace_PyTorch |   addInPlace |         16 |   float32 |        cpu |    409,414.47 μs |      1,567.731 μs |      85.933 μs |  1.00 |    0.00 |      Yes |
|            addInPlace_TorchSharp |   addInPlace |         16 |   float32 |        cpu |    357,585.47 μs |    756,247.524 μs |  41,452.479 μs |  0.87 |    0.10 |       No |
|       addInPlace_RawTensor_Torch |   addInPlace |         16 |   float32 |        cpu |    599,231.43 μs |    202,882.825 μs |  11,120.692 μs |  1.46 |    0.03 |       No |
|          addInPlace_Tensor_Torch |   addInPlace |         16 |   float32 |        cpu |  1,052,050.50 μs |    224,287.636 μs |  12,293.962 μs |  2.57 |    0.03 |       No |
|   addInPlace_RawTensor_Reference |   addInPlace |         16 |   float32 |        cpu |     13,203.48 μs |      2,060.142 μs |     112.923 μs |  0.03 |    0.00 |       No |
|      addInPlace_Tensor_Reference |   addInPlace |         16 |   float32 |        cpu |    170,016.87 μs |      9,922.113 μs |     543.864 μs |  0.42 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                   matmul_PyTorch |       matmul |         16 |   float32 |        cpu |               NA |                NA |             NA |     ? |       ? |      Yes |
|                matmul_TorchSharp |       matmul |         16 |   float32 |        cpu |     53,518.47 μs |     22,601.252 μs |   1,238.851 μs |     ? |       ? |       No |
|           matmul_RawTensor_Torch |       matmul |         16 |   float32 |        cpu |     59,325.90 μs |     21,305.933 μs |   1,167.850 μs |     ? |       ? |       No |
|              matmul_Tensor_Torch |       matmul |         16 |   float32 |        cpu |     95,296.23 μs |     12,905.581 μs |     707.398 μs |     ? |       ? |       No |
|       matmul_RawTensor_Reference |       matmul |         16 |   float32 |        cpu |     53,630.00 μs |     10,482.371 μs |     574.574 μs |     ? |       ? |       No |
|          matmul_Tensor_Reference |       matmul |         16 |   float32 |        cpu |     82,428.23 μs |    233,902.880 μs |  12,821.006 μs |     ? |       ? |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|              **fromCpuData_PyTorch** |  **fromCpuData** |         **16** |   **float32** |       **cuda** |  **3,459,695.90 μs** |      **5,825.486 μs** |     **319.315 μs** | **1.000** |    **0.00** |      **Yes** |
|           fromCpuData_TorchSharp |  fromCpuData |         16 |   float32 |       cuda |    156,730.13 μs |     53,154.346 μs |   2,913.569 μs | 0.045 |    0.00 |       No |
|      fromCpuData_RawTensor_Torch |  fromCpuData |         16 |   float32 |       cuda |  2,851,550.30 μs |    305,260.950 μs |  16,732.383 μs | 0.824 |    0.00 |       No |
|         fromCpuData_Tensor_Torch |  fromCpuData |         16 |   float32 |       cuda |  3,125,831.07 μs |  6,102,489.107 μs | 334,498.020 μs | 0.904 |    0.10 |       No |
|  fromCpuData_RawTensor_Reference |  fromCpuData |         16 |   float32 |       cuda |      2,175.03 μs |        440.510 μs |      24.146 μs | 0.001 |    0.00 |       No |
|     fromCpuData_Tensor_Reference |  fromCpuData |         16 |   float32 |       cuda |      3,827.20 μs |      2,826.353 μs |     154.922 μs | 0.001 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                    zeros_PyTorch |        zeros |         16 |   float32 |       cuda |  5,047,320.97 μs |      5,661.278 μs |     310.314 μs | 1.000 |    0.00 |      Yes |
|                 zeros_TorchSharp |        zeros |         16 |   float32 |       cuda |  2,927,802.83 μs |  1,182,605.690 μs |  64,822.608 μs | 0.580 |    0.01 |       No |
|            zeros_RawTensor_Torch |        zeros |         16 |   float32 |       cuda |  3,334,509.17 μs |  2,354,180.550 μs | 129,040.580 μs | 0.661 |    0.03 |       No |
|               zeros_Tensor_Torch |        zeros |         16 |   float32 |       cuda |  3,020,353.27 μs |    877,531.171 μs |  48,100.445 μs | 0.598 |    0.01 |       No |
|        zeros_RawTensor_Reference |        zeros |         16 |   float32 |       cuda |     11,632.18 μs |      1,615.279 μs |      88.539 μs | 0.002 |    0.00 |       No |
|           zeros_Tensor_Reference |        zeros |         16 |   float32 |       cuda |     12,949.33 μs |      6,031.445 μs |     330.604 μs | 0.003 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                     ones_PyTorch |         ones |         16 |   float32 |       cuda |  5,078,721.50 μs |      2,051.286 μs |     112.438 μs | 1.000 |    0.00 |      Yes |
|                  ones_TorchSharp |         ones |         16 |   float32 |       cuda |  2,997,636.53 μs |    981,423.124 μs |  53,795.113 μs | 0.590 |    0.01 |       No |
|             ones_RawTensor_Torch |         ones |         16 |   float32 |       cuda |  2,972,956.43 μs |    208,927.789 μs |  11,452.037 μs | 0.585 |    0.00 |       No |
|                ones_Tensor_Torch |         ones |         16 |   float32 |       cuda |  3,184,479.60 μs |    914,391.746 μs |  50,120.897 μs | 0.627 |    0.01 |       No |
|         ones_RawTensor_Reference |         ones |         16 |   float32 |       cuda |     12,883.35 μs |      4,526.139 μs |     248.093 μs | 0.003 |    0.00 |       No |
|            ones_Tensor_Reference |         ones |         16 |   float32 |       cuda |     14,494.18 μs |      4,911.815 μs |     269.233 μs | 0.003 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                     rand_PyTorch |         rand |         16 |   float32 |       cuda |  5,406,505.03 μs |      7,401.877 μs |     405.722 μs | 1.000 |    0.00 |      Yes |
|                  rand_TorchSharp |         rand |         16 |   float32 |       cuda |  3,045,404.10 μs |  2,211,711.631 μs | 121,231.378 μs | 0.563 |    0.02 |       No |
|             rand_RawTensor_Torch |         rand |         16 |   float32 |       cuda |  3,212,008.87 μs |    446,037.659 μs |  24,448.829 μs | 0.594 |    0.00 |       No |
|                rand_Tensor_Torch |         rand |         16 |   float32 |       cuda |  2,993,190.70 μs |  1,455,268.730 μs |  79,768.190 μs | 0.554 |    0.01 |       No |
|         rand_RawTensor_Reference |         rand |         16 |   float32 |       cuda |     44,897.67 μs |      5,944.985 μs |     325.865 μs | 0.008 |    0.00 |       No |
|            rand_Tensor_Reference |         rand |         16 |   float32 |       cuda |     63,101.87 μs |    156,359.024 μs |   8,570.566 μs | 0.012 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                 addition_PyTorch |     addition |         16 |   float32 |       cuda |  3,229,231.73 μs |        745.707 μs |      40.875 μs | 1.000 |    0.00 |      Yes |
|              addition_TorchSharp |     addition |         16 |   float32 |       cuda |  2,465,589.33 μs |    137,395.044 μs |   7,531.086 μs | 0.764 |    0.00 |       No |
|         addition_RawTensor_Torch |     addition |         16 |   float32 |       cuda |  2,619,163.93 μs |  1,590,965.553 μs |  87,206.190 μs | 0.811 |    0.03 |       No |
|            addition_Tensor_Torch |     addition |         16 |   float32 |       cuda |  3,531,617.20 μs |    898,578.580 μs |  49,254.124 μs | 1.094 |    0.02 |       No |
|     addition_RawTensor_Reference |     addition |         16 |   float32 |       cuda |     13,832.04 μs |      3,392.192 μs |     185.937 μs | 0.004 |    0.00 |       No |
|        addition_Tensor_Reference |     addition |         16 |   float32 |       cuda |    170,988.63 μs |     23,784.222 μs |   1,303.693 μs | 0.053 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                addScalar_PyTorch |    addScalar |         16 |   float32 |       cuda |  4,212,660.47 μs |      9,397.417 μs |     515.104 μs | 1.000 |    0.00 |      Yes |
|             addScalar_TorchSharp |    addScalar |         16 |   float32 |       cuda |  3,328,464.10 μs |  2,911,285.359 μs | 159,577.374 μs | 0.790 |    0.04 |       No |
|        addScalar_RawTensor_Torch |    addScalar |         16 |   float32 |       cuda | 17,783,628.73 μs |  4,906,150.818 μs | 268,922.681 μs | 4.221 |    0.06 |       No |
|           addScalar_Tensor_Torch |    addScalar |         16 |   float32 |       cuda | 20,092,143.30 μs | 14,846,470.927 μs | 813,785.169 μs | 4.769 |    0.19 |       No |
|    addScalar_RawTensor_Reference |    addScalar |         16 |   float32 |       cuda |      8,277.84 μs |      1,833.023 μs |     100.474 μs | 0.002 |    0.00 |       No |
|       addScalar_Tensor_Reference |    addScalar |         16 |   float32 |       cuda |    162,670.73 μs |     19,759.736 μs |   1,083.098 μs | 0.039 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|             addWithAlpha_PyTorch | addWithAlpha |         16 |   float32 |       cuda |  1,873,494.03 μs |      3,358.735 μs |     184.104 μs |  1.00 |    0.00 |      Yes |
|          addWithAlpha_TorchSharp | addWithAlpha |         16 |   float32 |       cuda |  2,534,033.13 μs |  3,333,765.993 μs | 182,734.963 μs |  1.35 |    0.10 |       No |
|     addWithAlpha_RawTensor_Torch | addWithAlpha |         16 |   float32 |       cuda |  5,958,947.53 μs |  1,472,734.254 μs |  80,725.534 μs |  3.18 |    0.04 |       No |
|        addWithAlpha_Tensor_Torch | addWithAlpha |         16 |   float32 |       cuda |  8,317,042.30 μs |  5,041,285.043 μs | 276,329.844 μs |  4.44 |    0.15 |       No |
| addWithAlpha_RawTensor_Reference | addWithAlpha |         16 |   float32 |       cuda |     21,162.68 μs |      2,153.867 μs |     118.061 μs |  0.01 |    0.00 |       No |
|    addWithAlpha_Tensor_Reference | addWithAlpha |         16 |   float32 |       cuda |    286,913.33 μs |     43,334.693 μs |   2,375.321 μs |  0.15 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|               addInPlace_PyTorch |   addInPlace |         16 |   float32 |       cuda |  1,643,633.77 μs |      6,003.669 μs |     329.081 μs | 1.000 |    0.00 |      Yes |
|            addInPlace_TorchSharp |   addInPlace |         16 |   float32 |       cuda |  1,545,815.53 μs |  3,445,769.597 μs | 188,874.259 μs | 0.940 |    0.11 |       No |
|       addInPlace_RawTensor_Torch |   addInPlace |         16 |   float32 |       cuda |  2,525,350.37 μs |  1,543,265.983 μs |  84,591.616 μs | 1.536 |    0.05 |       No |
|          addInPlace_Tensor_Torch |   addInPlace |         16 |   float32 |       cuda |  3,557,597.23 μs |    232,895.902 μs |  12,765.810 μs | 2.164 |    0.01 |       No |
|   addInPlace_RawTensor_Reference |   addInPlace |         16 |   float32 |       cuda |     13,727.94 μs |      3,429.858 μs |     188.002 μs | 0.008 |    0.00 |       No |
|      addInPlace_Tensor_Reference |   addInPlace |         16 |   float32 |       cuda |    162,996.60 μs |     28,318.658 μs |   1,552.241 μs | 0.099 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                   matmul_PyTorch |       matmul |         16 |   float32 |       cuda |               NA |                NA |             NA |     ? |       ? |      Yes |
|                matmul_TorchSharp |       matmul |         16 |   float32 |       cuda |    360,973.03 μs |    432,641.812 μs |  23,714.558 μs |     ? |       ? |       No |
|           matmul_RawTensor_Torch |       matmul |         16 |   float32 |       cuda |    351,507.23 μs |     48,047.474 μs |   2,633.644 μs |     ? |       ? |       No |
|              matmul_Tensor_Torch |       matmul |         16 |   float32 |       cuda |    482,894.43 μs |    118,221.148 μs |   6,480.100 μs |     ? |       ? |       No |
|       matmul_RawTensor_Reference |       matmul |         16 |   float32 |       cuda |     51,777.78 μs |      2,433.950 μs |     133.413 μs |     ? |       ? |       No |
|          matmul_Tensor_Reference |       matmul |         16 |   float32 |       cuda |     69,496.83 μs |     14,133.791 μs |     774.721 μs |     ? |       ? |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|              **fromCpuData_PyTorch** |  **fromCpuData** |         **16** |   **float64** |        **cpu** |    **535,500.63 μs** |      **4,684.889 μs** |     **256.795 μs** | **1.000** |    **0.00** |      **Yes** |
|           fromCpuData_TorchSharp |  fromCpuData |         16 |   float64 |        cpu |    163,652.10 μs |    109,494.656 μs |   6,001.772 μs | 0.306 |    0.01 |       No |
|      fromCpuData_RawTensor_Torch |  fromCpuData |         16 |   float64 |        cpu |    287,213.63 μs |    256,016.262 μs |  14,033.115 μs | 0.536 |    0.03 |       No |
|         fromCpuData_Tensor_Torch |  fromCpuData |         16 |   float64 |        cpu |    298,639.03 μs |    384,370.636 μs |  21,068.652 μs | 0.558 |    0.04 |       No |
|  fromCpuData_RawTensor_Reference |  fromCpuData |         16 |   float64 |        cpu |      2,227.06 μs |        496.127 μs |      27.194 μs | 0.004 |    0.00 |       No |
|     fromCpuData_Tensor_Reference |  fromCpuData |         16 |   float64 |        cpu |      4,499.01 μs |        357.613 μs |      19.602 μs | 0.008 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                    zeros_PyTorch |        zeros |         16 |   float64 |        cpu |  1,652,607.83 μs |      4,223.138 μs |     231.484 μs | 1.000 |    0.00 |      Yes |
|                 zeros_TorchSharp |        zeros |         16 |   float64 |        cpu |    557,649.47 μs |    151,139.386 μs |   8,284.460 μs | 0.337 |    0.01 |       No |
|            zeros_RawTensor_Torch |        zeros |         16 |   float64 |        cpu |    642,851.87 μs |    127,192.783 μs |   6,971.866 μs | 0.389 |    0.00 |       No |
|               zeros_Tensor_Torch |        zeros |         16 |   float64 |        cpu |    608,071.27 μs |    146,411.695 μs |   8,025.319 μs | 0.368 |    0.00 |       No |
|        zeros_RawTensor_Reference |        zeros |         16 |   float64 |        cpu |     12,723.25 μs |      3,223.247 μs |     176.677 μs | 0.008 |    0.00 |       No |
|           zeros_Tensor_Reference |        zeros |         16 |   float64 |        cpu |     13,611.97 μs |      3,920.724 μs |     214.908 μs | 0.008 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                     ones_PyTorch |         ones |         16 |   float64 |        cpu |  1,659,409.03 μs |      3,848.563 μs |     210.953 μs | 1.000 |    0.00 |      Yes |
|                  ones_TorchSharp |         ones |         16 |   float64 |        cpu |    561,688.30 μs |    323,115.408 μs |  17,711.046 μs | 0.338 |    0.01 |       No |
|             ones_RawTensor_Torch |         ones |         16 |   float64 |        cpu |    644,998.00 μs |    145,843.525 μs |   7,994.176 μs | 0.389 |    0.00 |       No |
|                ones_Tensor_Torch |         ones |         16 |   float64 |        cpu |    653,837.73 μs |    358,374.726 μs |  19,643.728 μs | 0.394 |    0.01 |       No |
|         ones_RawTensor_Reference |         ones |         16 |   float64 |        cpu |     13,911.71 μs |      3,849.153 μs |     210.985 μs | 0.008 |    0.00 |       No |
|            ones_Tensor_Reference |         ones |         16 |   float64 |        cpu |     15,695.74 μs |      2,770.429 μs |     151.857 μs | 0.009 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                     rand_PyTorch |         rand |         16 |   float64 |        cpu |  1,971,390.90 μs |      4,098.455 μs |     224.650 μs |  1.00 |    0.00 |      Yes |
|                  rand_TorchSharp |         rand |         16 |   float64 |        cpu |    642,802.43 μs |    185,449.618 μs |  10,165.119 μs |  0.33 |    0.01 |       No |
|             rand_RawTensor_Torch |         rand |         16 |   float64 |        cpu |    749,571.10 μs |    196,329.250 μs |  10,761.469 μs |  0.38 |    0.01 |       No |
|                rand_Tensor_Torch |         rand |         16 |   float64 |        cpu |    768,639.83 μs |    175,511.179 μs |   9,620.360 μs |  0.39 |    0.00 |       No |
|         rand_RawTensor_Reference |         rand |         16 |   float64 |        cpu |     46,863.72 μs |     15,591.745 μs |     854.636 μs |  0.02 |    0.00 |       No |
|            rand_Tensor_Reference |         rand |         16 |   float64 |        cpu |     46,001.59 μs |     12,458.142 μs |     682.873 μs |  0.02 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                 addition_PyTorch |     addition |         16 |   float64 |        cpu |    754,385.33 μs |      2,473.454 μs |     135.578 μs |  1.00 |    0.00 |      Yes |
|              addition_TorchSharp |     addition |         16 |   float64 |        cpu |    519,183.60 μs |     34,866.834 μs |   1,911.169 μs |  0.69 |    0.00 |       No |
|         addition_RawTensor_Torch |     addition |         16 |   float64 |        cpu |    564,843.80 μs |    207,549.609 μs |  11,376.494 μs |  0.75 |    0.02 |       No |
|            addition_Tensor_Torch |     addition |         16 |   float64 |        cpu |  1,026,632.00 μs |    289,936.182 μs |  15,892.380 μs |  1.36 |    0.02 |       No |
|     addition_RawTensor_Reference |     addition |         16 |   float64 |        cpu |     13,812.70 μs |      3,328.123 μs |     182.426 μs |  0.02 |    0.00 |       No |
|        addition_Tensor_Reference |     addition |         16 |   float64 |        cpu |    164,076.03 μs |     54,552.328 μs |   2,990.197 μs |  0.22 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                addScalar_PyTorch |    addScalar |         16 |   float64 |        cpu |  1,966,532.97 μs |      5,847.522 μs |     320.522 μs | 1.000 |    0.00 |      Yes |
|             addScalar_TorchSharp |    addScalar |         16 |   float64 |        cpu |  1,523,719.03 μs |    224,725.489 μs |  12,317.962 μs | 0.775 |    0.01 |       No |
|        addScalar_RawTensor_Torch |    addScalar |         16 |   float64 |        cpu |  1,812,391.77 μs |    197,509.416 μs |  10,826.158 μs | 0.922 |    0.01 |       No |
|           addScalar_Tensor_Torch |    addScalar |         16 |   float64 |        cpu |  2,493,344.20 μs |    324,977.091 μs |  17,813.091 μs | 1.268 |    0.01 |       No |
|    addScalar_RawTensor_Reference |    addScalar |         16 |   float64 |        cpu |      8,414.18 μs |      2,089.655 μs |     114.541 μs | 0.004 |    0.00 |       No |
|       addScalar_Tensor_Reference |    addScalar |         16 |   float64 |        cpu |    157,822.00 μs |     22,119.637 μs |   1,212.452 μs | 0.080 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|             addWithAlpha_PyTorch | addWithAlpha |         16 |   float64 |        cpu |    482,417.53 μs |      3,669.159 μs |     201.119 μs |  1.00 |    0.00 |      Yes |
|          addWithAlpha_TorchSharp | addWithAlpha |         16 |   float64 |        cpu |    479,211.97 μs |    218,509.239 μs |  11,977.229 μs |  0.99 |    0.03 |       No |
|     addWithAlpha_RawTensor_Torch | addWithAlpha |         16 |   float64 |        cpu |  1,307,303.10 μs |    467,727.628 μs |  25,637.730 μs |  2.71 |    0.05 |       No |
|        addWithAlpha_Tensor_Torch | addWithAlpha |         16 |   float64 |        cpu |  2,145,558.33 μs |    502,958.296 μs |  27,568.841 μs |  4.45 |    0.06 |       No |
| addWithAlpha_RawTensor_Reference | addWithAlpha |         16 |   float64 |        cpu |     21,635.32 μs |      3,280.656 μs |     179.824 μs |  0.04 |    0.00 |       No |
|    addWithAlpha_Tensor_Reference | addWithAlpha |         16 |   float64 |        cpu |    278,553.17 μs |     19,197.899 μs |   1,052.302 μs |  0.58 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|               addInPlace_PyTorch |   addInPlace |         16 |   float64 |        cpu |    385,296.70 μs |      2,054.158 μs |     112.595 μs |  1.00 |    0.00 |      Yes |
|            addInPlace_TorchSharp |   addInPlace |         16 |   float64 |        cpu |    302,476.47 μs |     48,922.370 μs |   2,681.600 μs |  0.79 |    0.01 |       No |
|       addInPlace_RawTensor_Torch |   addInPlace |         16 |   float64 |        cpu |    564,353.30 μs |     49,217.957 μs |   2,697.802 μs |  1.46 |    0.01 |       No |
|          addInPlace_Tensor_Torch |   addInPlace |         16 |   float64 |        cpu |  1,036,460.53 μs |    610,782.629 μs |  33,479.057 μs |  2.69 |    0.09 |       No |
|   addInPlace_RawTensor_Reference |   addInPlace |         16 |   float64 |        cpu |     13,566.56 μs |      4,648.433 μs |     254.796 μs |  0.04 |    0.00 |       No |
|      addInPlace_Tensor_Reference |   addInPlace |         16 |   float64 |        cpu |    163,541.47 μs |     15,441.746 μs |     846.414 μs |  0.42 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                   matmul_PyTorch |       matmul |         16 |   float64 |        cpu |               NA |                NA |             NA |     ? |       ? |      Yes |
|                matmul_TorchSharp |       matmul |         16 |   float64 |        cpu |     55,718.33 μs |      7,870.213 μs |     431.393 μs |     ? |       ? |       No |
|           matmul_RawTensor_Torch |       matmul |         16 |   float64 |        cpu |     58,972.26 μs |     24,926.595 μs |   1,366.311 μs |     ? |       ? |       No |
|              matmul_Tensor_Torch |       matmul |         16 |   float64 |        cpu |     95,331.73 μs |      6,455.326 μs |     353.838 μs |     ? |       ? |       No |
|       matmul_RawTensor_Reference |       matmul |         16 |   float64 |        cpu |     53,992.17 μs |     30,966.435 μs |   1,697.375 μs |     ? |       ? |       No |
|          matmul_Tensor_Reference |       matmul |         16 |   float64 |        cpu |     66,584.43 μs |      8,363.798 μs |     458.448 μs |     ? |       ? |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|              **fromCpuData_PyTorch** |  **fromCpuData** |         **16** |   **float64** |       **cuda** |  **3,515,390.57 μs** |      **7,086.431 μs** |     **388.431 μs** | **1.000** |    **0.00** |      **Yes** |
|           fromCpuData_TorchSharp |  fromCpuData |         16 |   float64 |       cuda |    157,621.10 μs |     30,845.836 μs |   1,690.764 μs | 0.045 |    0.00 |       No |
|      fromCpuData_RawTensor_Torch |  fromCpuData |         16 |   float64 |       cuda |  2,950,240.27 μs |  2,882,710.662 μs | 158,011.099 μs | 0.839 |    0.04 |       No |
|         fromCpuData_Tensor_Torch |  fromCpuData |         16 |   float64 |       cuda |  3,273,576.40 μs |  1,843,578.879 μs | 101,052.779 μs | 0.931 |    0.03 |       No |
|  fromCpuData_RawTensor_Reference |  fromCpuData |         16 |   float64 |       cuda |      2,213.83 μs |        464.857 μs |      25.480 μs | 0.001 |    0.00 |       No |
|     fromCpuData_Tensor_Reference |  fromCpuData |         16 |   float64 |       cuda |      4,497.19 μs |        423.143 μs |      23.194 μs | 0.001 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                    zeros_PyTorch |        zeros |         16 |   float64 |       cuda |  5,145,423.67 μs |      5,795.224 μs |     317.656 μs | 1.000 |    0.00 |      Yes |
|                 zeros_TorchSharp |        zeros |         16 |   float64 |       cuda |  2,644,617.10 μs |    237,813.767 μs |  13,035.375 μs | 0.514 |    0.00 |       No |
|            zeros_RawTensor_Torch |        zeros |         16 |   float64 |       cuda |  3,146,628.63 μs |  1,053,644.836 μs |  57,753.829 μs | 0.612 |    0.01 |       No |
|               zeros_Tensor_Torch |        zeros |         16 |   float64 |       cuda |  2,801,843.17 μs |    824,523.413 μs |  45,194.911 μs | 0.545 |    0.01 |       No |
|        zeros_RawTensor_Reference |        zeros |         16 |   float64 |       cuda |     12,671.82 μs |      2,297.505 μs |     125.934 μs | 0.002 |    0.00 |       No |
|           zeros_Tensor_Reference |        zeros |         16 |   float64 |       cuda |     14,004.20 μs |      1,165.390 μs |      63.879 μs | 0.003 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                     ones_PyTorch |         ones |         16 |   float64 |       cuda |  5,033,510.83 μs |      6,488.136 μs |     355.637 μs | 1.000 |    0.00 |      Yes |
|                  ones_TorchSharp |         ones |         16 |   float64 |       cuda |  2,841,282.87 μs |  1,980,837.544 μs | 108,576.390 μs | 0.564 |    0.02 |       No |
|             ones_RawTensor_Torch |         ones |         16 |   float64 |       cuda |  2,791,677.37 μs |    145,624.726 μs |   7,982.183 μs | 0.555 |    0.00 |       No |
|                ones_Tensor_Torch |         ones |         16 |   float64 |       cuda |  2,910,004.13 μs |  1,951,964.425 μs | 106,993.757 μs | 0.578 |    0.02 |       No |
|         ones_RawTensor_Reference |         ones |         16 |   float64 |       cuda |     14,082.36 μs |      5,179.890 μs |     283.927 μs | 0.003 |    0.00 |       No |
|            ones_Tensor_Reference |         ones |         16 |   float64 |       cuda |     15,150.02 μs |      1,061.765 μs |      58.199 μs | 0.003 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                     rand_PyTorch |         rand |         16 |   float64 |       cuda |  5,364,539.67 μs |      8,894.641 μs |     487.545 μs | 1.000 |    0.00 |      Yes |
|                  rand_TorchSharp |         rand |         16 |   float64 |       cuda |  2,748,008.13 μs |    276,922.575 μs |  15,179.061 μs | 0.512 |    0.00 |       No |
|             rand_RawTensor_Torch |         rand |         16 |   float64 |       cuda |  2,953,668.97 μs |    569,795.279 μs |  31,232.402 μs | 0.551 |    0.01 |       No |
|                rand_Tensor_Torch |         rand |         16 |   float64 |       cuda |  2,969,405.70 μs |    827,007.555 μs |  45,331.075 μs | 0.554 |    0.01 |       No |
|         rand_RawTensor_Reference |         rand |         16 |   float64 |       cuda |     47,470.07 μs |     24,641.411 μs |   1,350.679 μs | 0.009 |    0.00 |       No |
|            rand_Tensor_Reference |         rand |         16 |   float64 |       cuda |     46,588.82 μs |     19,987.184 μs |   1,095.565 μs | 0.009 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                 addition_PyTorch |     addition |         16 |   float64 |       cuda |  3,145,841.07 μs |      5,019.372 μs |     275.129 μs | 1.000 |    0.00 |      Yes |
|              addition_TorchSharp |     addition |         16 |   float64 |       cuda |  2,425,123.10 μs |  1,533,797.559 μs |  84,072.620 μs | 0.771 |    0.03 |       No |
|         addition_RawTensor_Torch |     addition |         16 |   float64 |       cuda |  2,588,562.27 μs |  1,311,197.133 μs |  71,871.139 μs | 0.823 |    0.02 |       No |
|            addition_Tensor_Torch |     addition |         16 |   float64 |       cuda |  3,667,172.07 μs |  1,740,759.946 μs |  95,416.927 μs | 1.166 |    0.03 |       No |
|     addition_RawTensor_Reference |     addition |         16 |   float64 |       cuda |     13,503.87 μs |      6,123.483 μs |     335.649 μs | 0.004 |    0.00 |       No |
|        addition_Tensor_Reference |     addition |         16 |   float64 |       cuda |    162,894.63 μs |     17,177.132 μs |     941.537 μs | 0.052 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                addScalar_PyTorch |    addScalar |         16 |   float64 |       cuda |  4,186,730.47 μs |      6,049.222 μs |     331.578 μs | 1.000 |    0.00 |      Yes |
|             addScalar_TorchSharp |    addScalar |         16 |   float64 |       cuda |  3,253,841.10 μs |  2,862,519.827 μs | 156,904.371 μs | 0.777 |    0.04 |       No |
|        addScalar_RawTensor_Torch |    addScalar |         16 |   float64 |       cuda | 18,507,552.83 μs | 12,902,722.428 μs | 707,241.755 μs | 4.421 |    0.17 |       No |
|           addScalar_Tensor_Torch |    addScalar |         16 |   float64 |       cuda | 19,541,734.50 μs |  1,731,577.726 μs |  94,913.618 μs | 4.668 |    0.02 |       No |
|    addScalar_RawTensor_Reference |    addScalar |         16 |   float64 |       cuda |      8,213.43 μs |      1,718.785 μs |      94.212 μs | 0.002 |    0.00 |       No |
|       addScalar_Tensor_Reference |    addScalar |         16 |   float64 |       cuda |    154,143.60 μs |     10,625.761 μs |     582.434 μs | 0.037 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|             addWithAlpha_PyTorch | addWithAlpha |         16 |   float64 |       cuda |  1,782,688.63 μs |      6,845.068 μs |     375.201 μs |  1.00 |    0.00 |      Yes |
|          addWithAlpha_TorchSharp | addWithAlpha |         16 |   float64 |       cuda |  2,602,427.90 μs |  1,637,895.227 μs |  89,778.564 μs |  1.46 |    0.05 |       No |
|     addWithAlpha_RawTensor_Torch | addWithAlpha |         16 |   float64 |       cuda |  5,975,230.87 μs |    465,817.630 μs |  25,533.036 μs |  3.35 |    0.01 |       No |
|        addWithAlpha_Tensor_Torch | addWithAlpha |         16 |   float64 |       cuda |  8,500,876.53 μs |    994,107.007 μs |  54,490.360 μs |  4.77 |    0.03 |       No |
| addWithAlpha_RawTensor_Reference | addWithAlpha |         16 |   float64 |       cuda |     23,084.00 μs |      8,345.446 μs |     457.442 μs |  0.01 |    0.00 |       No |
|    addWithAlpha_Tensor_Reference | addWithAlpha |         16 |   float64 |       cuda |    281,638.07 μs |     15,054.728 μs |     825.200 μs |  0.16 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|               addInPlace_PyTorch |   addInPlace |         16 |   float64 |       cuda |  1,600,519.73 μs |      5,430.031 μs |     297.638 μs | 1.000 |    0.00 |      Yes |
|            addInPlace_TorchSharp |   addInPlace |         16 |   float64 |       cuda |  1,554,235.00 μs |  1,235,732.180 μs |  67,734.651 μs | 0.971 |    0.04 |       No |
|       addInPlace_RawTensor_Torch |   addInPlace |         16 |   float64 |       cuda |  2,672,995.20 μs |    552,913.749 μs |  30,307.068 μs | 1.670 |    0.02 |       No |
|          addInPlace_Tensor_Torch |   addInPlace |         16 |   float64 |       cuda |  3,597,388.40 μs |    469,026.927 μs |  25,708.949 μs | 2.248 |    0.02 |       No |
|   addInPlace_RawTensor_Reference |   addInPlace |         16 |   float64 |       cuda |     13,848.36 μs |      2,548.601 μs |     139.697 μs | 0.009 |    0.00 |       No |
|      addInPlace_Tensor_Reference |   addInPlace |         16 |   float64 |       cuda |    160,778.37 μs |     31,877.984 μs |   1,747.340 μs | 0.100 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                   matmul_PyTorch |       matmul |         16 |   float64 |       cuda |               NA |                NA |             NA |     ? |       ? |      Yes |
|                matmul_TorchSharp |       matmul |         16 |   float64 |       cuda |    277,441.37 μs |     18,085.001 μs |     991.300 μs |     ? |       ? |       No |
|           matmul_RawTensor_Torch |       matmul |         16 |   float64 |       cuda |    310,525.77 μs |    439,904.760 μs |  24,112.664 μs |     ? |       ? |       No |
|              matmul_Tensor_Torch |       matmul |         16 |   float64 |       cuda |    370,822.57 μs |     48,244.985 μs |   2,644.470 μs |     ? |       ? |       No |
|       matmul_RawTensor_Reference |       matmul |         16 |   float64 |       cuda |     52,887.35 μs |      5,800.625 μs |     317.952 μs |     ? |       ? |       No |
|          matmul_Tensor_Reference |       matmul |         16 |   float64 |       cuda |     68,331.24 μs |     35,310.608 μs |   1,935.494 μs |     ? |       ? |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|              **fromCpuData_PyTorch** |  **fromCpuData** |         **16** |     **int32** |        **cpu** |    **505,107.40 μs** |        **701.402 μs** |      **38.446 μs** | **1.000** |    **0.00** |      **Yes** |
|           fromCpuData_TorchSharp |  fromCpuData |         16 |     int32 |        cpu |    159,996.57 μs |    102,012.153 μs |   5,591.630 μs | 0.317 |    0.01 |       No |
|      fromCpuData_RawTensor_Torch |  fromCpuData |         16 |     int32 |        cpu |    260,850.13 μs |     96,163.698 μs |   5,271.057 μs | 0.516 |    0.01 |       No |
|         fromCpuData_Tensor_Torch |  fromCpuData |         16 |     int32 |        cpu |    265,194.57 μs |    197,173.329 μs |  10,807.736 μs | 0.525 |    0.02 |       No |
|  fromCpuData_RawTensor_Reference |  fromCpuData |         16 |     int32 |        cpu |      2,114.78 μs |        424.350 μs |      23.260 μs | 0.004 |    0.00 |       No |
|     fromCpuData_Tensor_Reference |  fromCpuData |         16 |     int32 |        cpu |      2,727.92 μs |        405.911 μs |      22.249 μs | 0.005 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                    zeros_PyTorch |        zeros |         16 |     int32 |        cpu |  1,616,290.07 μs |      1,928.295 μs |     105.696 μs | 1.000 |    0.00 |      Yes |
|                 zeros_TorchSharp |        zeros |         16 |     int32 |        cpu |    568,753.40 μs |     77,182.531 μs |   4,230.635 μs | 0.352 |    0.00 |       No |
|            zeros_RawTensor_Torch |        zeros |         16 |     int32 |        cpu |    606,102.67 μs |    134,824.084 μs |   7,390.163 μs | 0.375 |    0.00 |       No |
|               zeros_Tensor_Torch |        zeros |         16 |     int32 |        cpu |    628,068.87 μs |    218,309.383 μs |  11,966.274 μs | 0.389 |    0.01 |       No |
|        zeros_RawTensor_Reference |        zeros |         16 |     int32 |        cpu |     11,633.56 μs |      1,735.640 μs |      95.136 μs | 0.007 |    0.00 |       No |
|           zeros_Tensor_Reference |        zeros |         16 |     int32 |        cpu |     12,900.68 μs |      1,089.892 μs |      59.741 μs | 0.008 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                     ones_PyTorch |         ones |         16 |     int32 |        cpu |  1,597,283.40 μs |      1,196.442 μs |      65.581 μs | 1.000 |    0.00 |      Yes |
|                  ones_TorchSharp |         ones |         16 |     int32 |        cpu |    590,600.57 μs |    600,595.031 μs |  32,920.640 μs | 0.370 |    0.02 |       No |
|             ones_RawTensor_Torch |         ones |         16 |     int32 |        cpu |    632,193.07 μs |     44,103.263 μs |   2,417.449 μs | 0.396 |    0.00 |       No |
|                ones_Tensor_Torch |         ones |         16 |     int32 |        cpu |    620,150.83 μs |     94,436.507 μs |   5,176.384 μs | 0.388 |    0.00 |       No |
|         ones_RawTensor_Reference |         ones |         16 |     int32 |        cpu |     14,420.22 μs |     16,901.258 μs |     926.415 μs | 0.009 |    0.00 |       No |
|            ones_Tensor_Reference |         ones |         16 |     int32 |        cpu |     14,333.29 μs |      2,764.678 μs |     151.541 μs | 0.009 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                     rand_PyTorch |         rand |         16 |     int32 |        cpu |               NA |                NA |             NA |     ? |       ? |      Yes |
|                  rand_TorchSharp |         rand |         16 |     int32 |        cpu |    650,981.53 μs |     34,133.504 μs |   1,870.973 μs |     ? |       ? |       No |
|             rand_RawTensor_Torch |         rand |         16 |     int32 |        cpu |               NA |                NA |             NA |     ? |       ? |       No |
|                rand_Tensor_Torch |         rand |         16 |     int32 |        cpu |               NA |                NA |             NA |     ? |       ? |       No |
|         rand_RawTensor_Reference |         rand |         16 |     int32 |        cpu |               NA |                NA |             NA |     ? |       ? |       No |
|            rand_Tensor_Reference |         rand |         16 |     int32 |        cpu |               NA |                NA |             NA |     ? |       ? |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                 addition_PyTorch |     addition |         16 |     int32 |        cpu |    714,746.30 μs |      8,648.692 μs |     474.064 μs |  1.00 |    0.00 |      Yes |
|              addition_TorchSharp |     addition |         16 |     int32 |        cpu |    501,892.20 μs |    311,500.132 μs |  17,074.373 μs |  0.70 |    0.02 |       No |
|         addition_RawTensor_Torch |     addition |         16 |     int32 |        cpu |    600,991.63 μs |    275,845.464 μs |  15,120.021 μs |  0.84 |    0.02 |       No |
|            addition_Tensor_Torch |     addition |         16 |     int32 |        cpu |  1,030,787.47 μs |    461,957.456 μs |  25,321.447 μs |  1.44 |    0.03 |       No |
|     addition_RawTensor_Reference |     addition |         16 |     int32 |        cpu |     14,105.93 μs |      6,833.676 μs |     374.577 μs |  0.02 |    0.00 |       No |
|        addition_Tensor_Reference |     addition |         16 |     int32 |        cpu |    168,971.03 μs |     63,402.837 μs |   3,475.323 μs |  0.24 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                addScalar_PyTorch |    addScalar |         16 |     int32 |        cpu |  1,863,595.60 μs |      8,103.907 μs |     444.202 μs |  1.00 |    0.00 |      Yes |
|             addScalar_TorchSharp |    addScalar |         16 |     int32 |        cpu |  1,485,909.50 μs |    588,461.952 μs |  32,255.585 μs |  0.80 |    0.02 |       No |
|        addScalar_RawTensor_Torch |    addScalar |         16 |     int32 |        cpu |  2,344,650.23 μs |    949,664.156 μs |  52,054.297 μs |  1.26 |    0.03 |       No |
|           addScalar_Tensor_Torch |    addScalar |         16 |     int32 |        cpu |  3,602,570.17 μs |    707,956.494 μs |  38,805.484 μs |  1.93 |    0.02 |       No |
|    addScalar_RawTensor_Reference |    addScalar |         16 |     int32 |        cpu |               NA |                NA |             NA |     ? |       ? |       No |
|       addScalar_Tensor_Reference |    addScalar |         16 |     int32 |        cpu |  3,886,736.73 μs |    384,979.036 μs |  21,102.000 μs |  2.09 |    0.01 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|             addWithAlpha_PyTorch | addWithAlpha |         16 |     int32 |        cpu |    485,678.80 μs |      6,917.148 μs |     379.152 μs |  1.00 |    0.00 |      Yes |
|          addWithAlpha_TorchSharp | addWithAlpha |         16 |     int32 |        cpu |    479,380.27 μs |     77,862.848 μs |   4,267.925 μs |  0.99 |    0.01 |       No |
|     addWithAlpha_RawTensor_Torch | addWithAlpha |         16 |     int32 |        cpu |  2,781,153.53 μs |    704,799.895 μs |  38,632.461 μs |  5.73 |    0.08 |       No |
|        addWithAlpha_Tensor_Torch | addWithAlpha |         16 |     int32 |        cpu |  4,621,609.10 μs |    959,403.670 μs |  52,588.152 μs |  9.52 |    0.11 |       No |
| addWithAlpha_RawTensor_Reference | addWithAlpha |         16 |     int32 |        cpu |               NA |                NA |             NA |     ? |       ? |       No |
|    addWithAlpha_Tensor_Reference | addWithAlpha |         16 |     int32 |        cpu |  7,743,113.30 μs |    102,627.914 μs |   5,625.382 μs | 15.94 |    0.02 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|               addInPlace_PyTorch |   addInPlace |         16 |     int32 |        cpu |    383,385.90 μs |      4,261.012 μs |     233.560 μs |  1.00 |    0.00 |      Yes |
|            addInPlace_TorchSharp |   addInPlace |         16 |     int32 |        cpu |    296,222.00 μs |    461,243.532 μs |  25,282.314 μs |  0.77 |    0.07 |       No |
|       addInPlace_RawTensor_Torch |   addInPlace |         16 |     int32 |        cpu |    565,930.27 μs |    295,716.990 μs |  16,209.246 μs |  1.48 |    0.04 |       No |
|          addInPlace_Tensor_Torch |   addInPlace |         16 |     int32 |        cpu |  1,039,076.73 μs |    317,807.986 μs |  17,420.128 μs |  2.71 |    0.05 |       No |
|   addInPlace_RawTensor_Reference |   addInPlace |         16 |     int32 |        cpu |     12,960.66 μs |      1,227.616 μs |      67.290 μs |  0.03 |    0.00 |       No |
|      addInPlace_Tensor_Reference |   addInPlace |         16 |     int32 |        cpu |    159,506.00 μs |     16,824.080 μs |     922.185 μs |  0.42 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                   matmul_PyTorch |       matmul |         16 |     int32 |        cpu |               NA |                NA |             NA |     ? |       ? |      Yes |
|                matmul_TorchSharp |       matmul |         16 |     int32 |        cpu |     47,993.43 μs |     28,473.263 μs |   1,560.716 μs |     ? |       ? |       No |
|           matmul_RawTensor_Torch |       matmul |         16 |     int32 |        cpu |     51,090.35 μs |      7,219.065 μs |     395.701 μs |     ? |       ? |       No |
|              matmul_Tensor_Torch |       matmul |         16 |     int32 |        cpu |     84,094.93 μs |     41,327.568 μs |   2,265.303 μs |     ? |       ? |       No |
|       matmul_RawTensor_Reference |       matmul |         16 |     int32 |        cpu |     53,460.19 μs |      9,510.161 μs |     521.284 μs |     ? |       ? |       No |
|          matmul_Tensor_Reference |       matmul |         16 |     int32 |        cpu |     68,263.16 μs |      9,028.259 μs |     494.869 μs |     ? |       ? |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|              **fromCpuData_PyTorch** |  **fromCpuData** |         **16** |     **int32** |       **cuda** |  **3,502,576.93 μs** |      **2,271.510 μs** |     **124.509 μs** | **1.000** |    **0.00** |      **Yes** |
|           fromCpuData_TorchSharp |  fromCpuData |         16 |     int32 |       cuda |    154,706.00 μs |     38,205.398 μs |   2,094.167 μs | 0.044 |    0.00 |       No |
|      fromCpuData_RawTensor_Torch |  fromCpuData |         16 |     int32 |       cuda |  3,587,197.60 μs |  9,262,706.852 μs | 507,720.218 μs | 1.024 |    0.14 |       No |
|         fromCpuData_Tensor_Torch |  fromCpuData |         16 |     int32 |       cuda |  2,893,057.70 μs |    128,611.920 μs |   7,049.653 μs | 0.826 |    0.00 |       No |
|  fromCpuData_RawTensor_Reference |  fromCpuData |         16 |     int32 |       cuda |      2,071.62 μs |        294.978 μs |      16.169 μs | 0.001 |    0.00 |       No |
|     fromCpuData_Tensor_Reference |  fromCpuData |         16 |     int32 |       cuda |      2,603.84 μs |        723.025 μs |      39.631 μs | 0.001 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                    zeros_PyTorch |        zeros |         16 |     int32 |       cuda |  5,051,765.03 μs |      7,666.166 μs |     420.208 μs | 1.000 |    0.00 |      Yes |
|                 zeros_TorchSharp |        zeros |         16 |     int32 |       cuda |  2,961,785.17 μs |  2,051,910.331 μs | 112,472.129 μs | 0.586 |    0.02 |       No |
|            zeros_RawTensor_Torch |        zeros |         16 |     int32 |       cuda |  2,792,807.80 μs |    125,621.247 μs |   6,885.724 μs | 0.553 |    0.00 |       No |
|               zeros_Tensor_Torch |        zeros |         16 |     int32 |       cuda |  2,885,267.43 μs |  1,084,384.019 μs |  59,438.747 μs | 0.571 |    0.01 |       No |
|        zeros_RawTensor_Reference |        zeros |         16 |     int32 |       cuda |     11,491.08 μs |      3,650.255 μs |     200.083 μs | 0.002 |    0.00 |       No |
|           zeros_Tensor_Reference |        zeros |         16 |     int32 |       cuda |     13,034.93 μs |      4,161.332 μs |     228.097 μs | 0.003 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                     ones_PyTorch |         ones |         16 |     int32 |       cuda |  4,971,443.20 μs |      8,198.878 μs |     449.408 μs | 1.000 |    0.00 |      Yes |
|                  ones_TorchSharp |         ones |         16 |     int32 |       cuda |  2,649,665.33 μs |    357,857.042 μs |  19,615.352 μs | 0.533 |    0.00 |       No |
|             ones_RawTensor_Torch |         ones |         16 |     int32 |       cuda |  3,055,592.57 μs |  1,354,337.598 μs |  74,235.814 μs | 0.615 |    0.01 |       No |
|                ones_Tensor_Torch |         ones |         16 |     int32 |       cuda |  2,840,099.37 μs |    567,154.906 μs |  31,087.674 μs | 0.571 |    0.01 |       No |
|         ones_RawTensor_Reference |         ones |         16 |     int32 |       cuda |     13,010.52 μs |     12,046.183 μs |     660.292 μs | 0.003 |    0.00 |       No |
|            ones_Tensor_Reference |         ones |         16 |     int32 |       cuda |     14,289.07 μs |      4,763.780 μs |     261.119 μs | 0.003 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                     rand_PyTorch |         rand |         16 |     int32 |       cuda |               NA |                NA |             NA |     ? |       ? |      Yes |
|                  rand_TorchSharp |         rand |         16 |     int32 |       cuda |  2,794,042.93 μs |    487,308.415 μs |  26,711.019 μs |     ? |       ? |       No |
|             rand_RawTensor_Torch |         rand |         16 |     int32 |       cuda |               NA |                NA |             NA |     ? |       ? |       No |
|                rand_Tensor_Torch |         rand |         16 |     int32 |       cuda |               NA |                NA |             NA |     ? |       ? |       No |
|         rand_RawTensor_Reference |         rand |         16 |     int32 |       cuda |               NA |                NA |             NA |     ? |       ? |       No |
|            rand_Tensor_Reference |         rand |         16 |     int32 |       cuda |               NA |                NA |             NA |     ? |       ? |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                 addition_PyTorch |     addition |         16 |     int32 |       cuda |  3,260,757.20 μs |      3,220.093 μs |     176.504 μs | 1.000 |    0.00 |      Yes |
|              addition_TorchSharp |     addition |         16 |     int32 |       cuda |  2,717,734.17 μs |  1,014,757.410 μs |  55,622.278 μs | 0.833 |    0.02 |       No |
|         addition_RawTensor_Torch |     addition |         16 |     int32 |       cuda |  2,660,094.00 μs |  2,252,461.215 μs | 123,465.000 μs | 0.816 |    0.04 |       No |
|            addition_Tensor_Torch |     addition |         16 |     int32 |       cuda |  3,495,033.10 μs |    304,193.743 μs |  16,673.886 μs | 1.072 |    0.01 |       No |
|     addition_RawTensor_Reference |     addition |         16 |     int32 |       cuda |     12,921.50 μs |      1,979.617 μs |     108.509 μs | 0.004 |    0.00 |       No |
|        addition_Tensor_Reference |     addition |         16 |     int32 |       cuda |    166,427.23 μs |     41,075.087 μs |   2,251.464 μs | 0.051 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                addScalar_PyTorch |    addScalar |         16 |     int32 |       cuda |  4,333,646.10 μs |      4,385.883 μs |     240.405 μs |  1.00 |    0.00 |      Yes |
|             addScalar_TorchSharp |    addScalar |         16 |     int32 |       cuda |  3,129,696.10 μs |    195,812.655 μs |  10,733.152 μs |  0.72 |    0.00 |       No |
|        addScalar_RawTensor_Torch |    addScalar |         16 |     int32 |       cuda | 19,416,757.00 μs | 15,554,074.138 μs | 852,571.288 μs |  4.48 |    0.20 |       No |
|           addScalar_Tensor_Torch |    addScalar |         16 |     int32 |       cuda | 25,088,036.47 μs | 10,317,306.317 μs | 565,526.373 μs |  5.79 |    0.13 |       No |
|    addScalar_RawTensor_Reference |    addScalar |         16 |     int32 |       cuda |               NA |                NA |             NA |     ? |       ? |       No |
|       addScalar_Tensor_Reference |    addScalar |         16 |     int32 |       cuda |  3,858,162.53 μs |    345,940.649 μs |  18,962.174 μs |  0.89 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|             addWithAlpha_PyTorch | addWithAlpha |         16 |     int32 |       cuda |  1,804,493.87 μs |      3,592.807 μs |     196.934 μs |  1.00 |    0.00 |      Yes |
|          addWithAlpha_TorchSharp | addWithAlpha |         16 |     int32 |       cuda |  2,201,832.60 μs |    238,233.841 μs |  13,058.401 μs |  1.22 |    0.01 |       No |
|     addWithAlpha_RawTensor_Torch | addWithAlpha |         16 |     int32 |       cuda |  6,866,064.27 μs |  3,888,309.717 μs | 213,131.376 μs |  3.80 |    0.12 |       No |
|        addWithAlpha_Tensor_Torch | addWithAlpha |         16 |     int32 |       cuda | 17,017,629.73 μs |  3,575,565.976 μs | 195,988.836 μs |  9.43 |    0.11 |       No |
| addWithAlpha_RawTensor_Reference | addWithAlpha |         16 |     int32 |       cuda |               NA |                NA |             NA |     ? |       ? |       No |
|    addWithAlpha_Tensor_Reference | addWithAlpha |         16 |     int32 |       cuda |  7,658,960.33 μs |  1,212,167.704 μs |  66,443.002 μs |  4.24 |    0.04 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|               addInPlace_PyTorch |   addInPlace |         16 |     int32 |       cuda |  1,613,768.97 μs |      7,546.610 μs |     413.655 μs | 1.000 |    0.00 |      Yes |
|            addInPlace_TorchSharp |   addInPlace |         16 |     int32 |       cuda |  1,438,337.70 μs |    872,825.089 μs |  47,842.488 μs | 0.891 |    0.03 |       No |
|       addInPlace_RawTensor_Torch |   addInPlace |         16 |     int32 |       cuda |  2,492,992.90 μs |  2,205,652.303 μs | 120,899.245 μs | 1.545 |    0.07 |       No |
|          addInPlace_Tensor_Torch |   addInPlace |         16 |     int32 |       cuda |  3,509,123.43 μs |    504,643.706 μs |  27,661.224 μs | 2.174 |    0.02 |       No |
|   addInPlace_RawTensor_Reference |   addInPlace |         16 |     int32 |       cuda |     13,444.98 μs |      4,294.263 μs |     235.383 μs | 0.008 |    0.00 |       No |
|      addInPlace_Tensor_Reference |   addInPlace |         16 |     int32 |       cuda |    168,168.13 μs |     35,050.142 μs |   1,921.217 μs | 0.104 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                   matmul_PyTorch |       matmul |         16 |     int32 |       cuda |               NA |                NA |             NA |     ? |       ? |      Yes |
|                matmul_TorchSharp |       matmul |         16 |     int32 |       cuda |               NA |                NA |             NA |     ? |       ? |       No |
|           matmul_RawTensor_Torch |       matmul |         16 |     int32 |       cuda |  2,005,014.10 μs |    962,806.305 μs |  52,774.662 μs |     ? |       ? |       No |
|              matmul_Tensor_Torch |       matmul |         16 |     int32 |       cuda |  1,991,506.87 μs |  1,417,194.225 μs |  77,681.198 μs |     ? |       ? |       No |
|       matmul_RawTensor_Reference |       matmul |         16 |     int32 |       cuda |     54,848.08 μs |     25,306.939 μs |   1,387.159 μs |     ? |       ? |       No |
|          matmul_Tensor_Reference |       matmul |         16 |     int32 |       cuda |     69,605.23 μs |     27,948.147 μs |   1,531.932 μs |     ? |       ? |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|              **fromCpuData_PyTorch** |  **fromCpuData** |       **2048** |   **float32** |        **cpu** |     **29,558.30 μs** |      **1,286.492 μs** |      **70.517 μs** | **1.000** |    **0.00** |      **Yes** |
|           fromCpuData_TorchSharp |  fromCpuData |       2048 |   float32 |        cpu |      1,637.47 μs |        529.148 μs |      29.004 μs | 0.055 |    0.00 |       No |
|      fromCpuData_RawTensor_Torch |  fromCpuData |       2048 |   float32 |        cpu |      4,027.56 μs |      1,425.664 μs |      78.145 μs | 0.136 |    0.00 |       No |
|         fromCpuData_Tensor_Torch |  fromCpuData |       2048 |   float32 |        cpu |      4,205.40 μs |        545.369 μs |      29.894 μs | 0.142 |    0.00 |       No |
|  fromCpuData_RawTensor_Reference |  fromCpuData |       2048 |   float32 |        cpu |        159.00 μs |          8.277 μs |       0.454 μs | 0.005 |    0.00 |       No |
|     fromCpuData_Tensor_Reference |  fromCpuData |       2048 |   float32 |        cpu |        175.25 μs |         81.999 μs |       4.495 μs | 0.006 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                    zeros_PyTorch |        zeros |       2048 |   float32 |        cpu |     15,563.99 μs |      1,949.629 μs |     106.866 μs |  1.00 |    0.00 |      Yes |
|                 zeros_TorchSharp |        zeros |       2048 |   float32 |        cpu |     19,578.48 μs |        676.166 μs |      37.063 μs |  1.26 |    0.01 |       No |
|            zeros_RawTensor_Torch |        zeros |       2048 |   float32 |        cpu |     10,881.15 μs |      3,701.343 μs |     202.883 μs |  0.70 |    0.01 |       No |
|               zeros_Tensor_Torch |        zeros |       2048 |   float32 |        cpu |     11,044.80 μs |      1,969.539 μs |     107.957 μs |  0.71 |    0.00 |       No |
|        zeros_RawTensor_Reference |        zeros |       2048 |   float32 |        cpu |      1,177.33 μs |         61.643 μs |       3.379 μs |  0.08 |    0.00 |       No |
|           zeros_Tensor_Reference |        zeros |       2048 |   float32 |        cpu |      1,210.00 μs |        489.994 μs |      26.858 μs |  0.08 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                     ones_PyTorch |         ones |       2048 |   float32 |        cpu |     14,572.60 μs |        311.421 μs |      17.070 μs |  1.00 |    0.00 |      Yes |
|                  ones_TorchSharp |         ones |       2048 |   float32 |        cpu |     19,107.39 μs |      6,300.004 μs |     345.324 μs |  1.31 |    0.03 |       No |
|             ones_RawTensor_Torch |         ones |       2048 |   float32 |        cpu |     10,342.07 μs |      2,150.050 μs |     117.851 μs |  0.71 |    0.01 |       No |
|                ones_Tensor_Torch |         ones |       2048 |   float32 |        cpu |     10,696.09 μs |      2,718.401 μs |     149.005 μs |  0.73 |    0.01 |       No |
|         ones_RawTensor_Reference |         ones |       2048 |   float32 |        cpu |      2,678.25 μs |        511.941 μs |      28.061 μs |  0.18 |    0.00 |       No |
|            ones_Tensor_Reference |         ones |       2048 |   float32 |        cpu |      2,693.71 μs |         84.168 μs |       4.614 μs |  0.18 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                     rand_PyTorch |         rand |       2048 |   float32 |        cpu |     31,592.95 μs |      1,911.099 μs |     104.754 μs |  1.00 |    0.00 |      Yes |
|                  rand_TorchSharp |         rand |       2048 |   float32 |        cpu |     53,646.88 μs |     10,027.522 μs |     549.642 μs |  1.70 |    0.02 |       No |
|             rand_RawTensor_Torch |         rand |       2048 |   float32 |        cpu |     27,535.63 μs |        964.574 μs |      52.872 μs |  0.87 |    0.00 |       No |
|                rand_Tensor_Torch |         rand |       2048 |   float32 |        cpu |     28,220.28 μs |      9,144.919 μs |     501.264 μs |  0.89 |    0.01 |       No |
|         rand_RawTensor_Reference |         rand |       2048 |   float32 |        cpu |     34,943.35 μs |      3,269.123 μs |     179.192 μs |  1.11 |    0.00 |       No |
|            rand_Tensor_Reference |         rand |       2048 |   float32 |        cpu |     32,666.03 μs |      7,308.029 μs |     400.578 μs |  1.03 |    0.01 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                 addition_PyTorch |     addition |       2048 |   float32 |        cpu |      9,679.31 μs |        703.191 μs |      38.544 μs |  1.00 |    0.00 |      Yes |
|              addition_TorchSharp |     addition |       2048 |   float32 |        cpu |     11,946.65 μs |      1,276.241 μs |      69.955 μs |  1.23 |    0.01 |       No |
|         addition_RawTensor_Torch |     addition |       2048 |   float32 |        cpu |     11,124.78 μs |      5,714.144 μs |     313.212 μs |  1.15 |    0.03 |       No |
|            addition_Tensor_Torch |     addition |       2048 |   float32 |        cpu |     14,926.15 μs |      5,141.143 μs |     281.803 μs |  1.54 |    0.03 |       No |
|     addition_RawTensor_Reference |     addition |       2048 |   float32 |        cpu |      6,504.38 μs |      1,258.090 μs |      68.960 μs |  0.67 |    0.01 |       No |
|        addition_Tensor_Reference |     addition |       2048 |   float32 |        cpu |      7,951.93 μs |      1,103.930 μs |      60.510 μs |  0.82 |    0.01 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                addScalar_PyTorch |    addScalar |       2048 |   float32 |        cpu |     19,542.46 μs |        418.278 μs |      22.927 μs |  1.00 |    0.00 |      Yes |
|             addScalar_TorchSharp |    addScalar |       2048 |   float32 |        cpu |     21,092.09 μs |      3,601.663 μs |     197.419 μs |  1.08 |    0.01 |       No |
|        addScalar_RawTensor_Torch |    addScalar |       2048 |   float32 |        cpu |     23,267.48 μs |      5,415.934 μs |     296.866 μs |  1.19 |    0.02 |       No |
|           addScalar_Tensor_Torch |    addScalar |       2048 |   float32 |        cpu |     28,874.86 μs |      3,660.535 μs |     200.646 μs |  1.48 |    0.01 |       No |
|    addScalar_RawTensor_Reference |    addScalar |       2048 |   float32 |        cpu |      2,625.91 μs |         57.322 μs |       3.142 μs |  0.13 |    0.00 |       No |
|       addScalar_Tensor_Reference |    addScalar |       2048 |   float32 |        cpu |      4,231.38 μs |        352.429 μs |      19.318 μs |  0.22 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|             addWithAlpha_PyTorch | addWithAlpha |       2048 |   float32 |        cpu |      6,694.54 μs |        456.071 μs |      24.999 μs |  1.00 |    0.00 |      Yes |
|          addWithAlpha_TorchSharp | addWithAlpha |       2048 |   float32 |        cpu |     12,080.58 μs |      4,549.358 μs |     249.366 μs |  1.80 |    0.03 |       No |
|     addWithAlpha_RawTensor_Torch | addWithAlpha |       2048 |   float32 |        cpu |     24,879.46 μs |     14,575.732 μs |     798.945 μs |  3.72 |    0.12 |       No |
|        addWithAlpha_Tensor_Torch | addWithAlpha |       2048 |   float32 |        cpu |     37,495.67 μs |     23,383.227 μs |   1,281.714 μs |  5.60 |    0.17 |       No |
| addWithAlpha_RawTensor_Reference | addWithAlpha |       2048 |   float32 |        cpu |      9,731.96 μs |      1,094.464 μs |      59.991 μs |  1.45 |    0.01 |       No |
|    addWithAlpha_Tensor_Reference | addWithAlpha |       2048 |   float32 |        cpu |     12,943.07 μs |      9,423.331 μs |     516.525 μs |  1.93 |    0.07 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|               addInPlace_PyTorch |   addInPlace |       2048 |   float32 |        cpu |      5,668.32 μs |        374.139 μs |      20.508 μs |  1.00 |    0.00 |      Yes |
|            addInPlace_TorchSharp |   addInPlace |       2048 |   float32 |        cpu |      4,958.71 μs |      2,117.885 μs |     116.088 μs |  0.87 |    0.02 |       No |
|       addInPlace_RawTensor_Torch |   addInPlace |       2048 |   float32 |        cpu |     10,986.31 μs |      1,910.514 μs |     104.722 μs |  1.94 |    0.02 |       No |
|          addInPlace_Tensor_Torch |   addInPlace |       2048 |   float32 |        cpu |     14,109.30 μs |      4,323.326 μs |     236.976 μs |  2.49 |    0.05 |       No |
|   addInPlace_RawTensor_Reference |   addInPlace |       2048 |   float32 |        cpu |      6,430.86 μs |      2,021.318 μs |     110.795 μs |  1.13 |    0.02 |       No |
|      addInPlace_Tensor_Reference |   addInPlace |       2048 |   float32 |        cpu |      7,858.12 μs |      1,738.240 μs |      95.279 μs |  1.39 |    0.01 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                   matmul_PyTorch |       matmul |       2048 |   float32 |        cpu |               NA |                NA |             NA |     ? |       ? |      Yes |
|                matmul_TorchSharp |       matmul |       2048 |   float32 |        cpu |      1,769.07 μs |        864.085 μs |      47.363 μs |     ? |       ? |       No |
|           matmul_RawTensor_Torch |       matmul |       2048 |   float32 |        cpu |      2,969.05 μs |      1,166.289 μs |      63.928 μs |     ? |       ? |       No |
|              matmul_Tensor_Torch |       matmul |       2048 |   float32 |        cpu |      3,523.65 μs |      1,338.821 μs |      73.385 μs |     ? |       ? |       No |
|       matmul_RawTensor_Reference |       matmul |       2048 |   float32 |        cpu |    211,015.70 μs |     35,085.619 μs |   1,923.161 μs |     ? |       ? |       No |
|          matmul_Tensor_Reference |       matmul |       2048 |   float32 |        cpu |    208,951.03 μs |     49,271.996 μs |   2,700.764 μs |     ? |       ? |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|              **fromCpuData_PyTorch** |  **fromCpuData** |       **2048** |   **float32** |       **cuda** |     **52,636.56 μs** |        **925.824 μs** |      **50.748 μs** | **1.000** |    **0.00** |      **Yes** |
|           fromCpuData_TorchSharp |  fromCpuData |       2048 |   float32 |       cuda |      1,207.33 μs |      2,772.690 μs |     151.980 μs | 0.023 |    0.00 |       No |
|      fromCpuData_RawTensor_Torch |  fromCpuData |       2048 |   float32 |       cuda |     24,411.00 μs |      8,064.316 μs |     442.032 μs | 0.464 |    0.01 |       No |
|         fromCpuData_Tensor_Torch |  fromCpuData |       2048 |   float32 |       cuda |     32,395.27 μs |        195.824 μs |      10.734 μs | 0.615 |    0.00 |       No |
|  fromCpuData_RawTensor_Reference |  fromCpuData |       2048 |   float32 |       cuda |        160.34 μs |         71.891 μs |       3.941 μs | 0.003 |    0.00 |       No |
|     fromCpuData_Tensor_Reference |  fromCpuData |       2048 |   float32 |       cuda |        175.29 μs |         46.721 μs |       2.561 μs | 0.003 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                    zeros_PyTorch |        zeros |       2048 |   float32 |       cuda |     39,413.13 μs |        728.377 μs |      39.925 μs |  1.00 |    0.00 |      Yes |
|                 zeros_TorchSharp |        zeros |       2048 |   float32 |       cuda |     24,202.03 μs |     16,098.244 μs |     882.399 μs |  0.61 |    0.02 |       No |
|            zeros_RawTensor_Torch |        zeros |       2048 |   float32 |       cuda |     22,097.37 μs |      2,805.847 μs |     153.798 μs |  0.56 |    0.00 |       No |
|               zeros_Tensor_Torch |        zeros |       2048 |   float32 |       cuda |     25,726.17 μs |     16,060.095 μs |     880.308 μs |  0.65 |    0.02 |       No |
|        zeros_RawTensor_Reference |        zeros |       2048 |   float32 |       cuda |      1,189.15 μs |         53.594 μs |       2.938 μs |  0.03 |    0.00 |       No |
|           zeros_Tensor_Reference |        zeros |       2048 |   float32 |       cuda |      1,212.39 μs |        272.244 μs |      14.923 μs |  0.03 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                     ones_PyTorch |         ones |       2048 |   float32 |       cuda |     38,554.89 μs |      2,320.249 μs |     127.181 μs |  1.00 |    0.00 |      Yes |
|                  ones_TorchSharp |         ones |       2048 |   float32 |       cuda |     23,281.23 μs |      5,806.931 μs |     318.297 μs |  0.60 |    0.01 |       No |
|             ones_RawTensor_Torch |         ones |       2048 |   float32 |       cuda |     22,058.93 μs |      6,372.794 μs |     349.314 μs |  0.57 |    0.01 |       No |
|                ones_Tensor_Torch |         ones |       2048 |   float32 |       cuda |     25,186.23 μs |     11,942.603 μs |     654.614 μs |  0.65 |    0.02 |       No |
|         ones_RawTensor_Reference |         ones |       2048 |   float32 |       cuda |      2,715.27 μs |        294.919 μs |      16.166 μs |  0.07 |    0.00 |       No |
|            ones_Tensor_Reference |         ones |       2048 |   float32 |       cuda |      2,788.25 μs |      1,247.647 μs |      68.388 μs |  0.07 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                     rand_PyTorch |         rand |       2048 |   float32 |       cuda |     40,522.09 μs |      1,344.397 μs |      73.691 μs |  1.00 |    0.00 |      Yes |
|                  rand_TorchSharp |         rand |       2048 |   float32 |       cuda |     22,561.83 μs |      2,711.281 μs |     148.614 μs |  0.56 |    0.00 |       No |
|             rand_RawTensor_Torch |         rand |       2048 |   float32 |       cuda |     24,826.87 μs |     36,131.500 μs |   1,980.489 μs |  0.61 |    0.05 |       No |
|                rand_Tensor_Torch |         rand |       2048 |   float32 |       cuda |     25,206.10 μs |     34,412.488 μs |   1,886.265 μs |  0.62 |    0.05 |       No |
|         rand_RawTensor_Reference |         rand |       2048 |   float32 |       cuda |     35,138.85 μs |     11,346.522 μs |     621.941 μs |  0.87 |    0.02 |       No |
|            rand_Tensor_Reference |         rand |       2048 |   float32 |       cuda |     32,695.09 μs |      4,275.316 μs |     234.345 μs |  0.81 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                 addition_PyTorch |     addition |       2048 |   float32 |       cuda |     25,502.45 μs |      1,347.812 μs |      73.878 μs |  1.00 |    0.00 |      Yes |
|              addition_TorchSharp |     addition |       2048 |   float32 |       cuda |     18,265.23 μs |      4,373.132 μs |     239.706 μs |  0.72 |    0.01 |       No |
|         addition_RawTensor_Torch |     addition |       2048 |   float32 |       cuda |     20,229.07 μs |      9,384.126 μs |     514.376 μs |  0.79 |    0.02 |       No |
|            addition_Tensor_Torch |     addition |       2048 |   float32 |       cuda |     33,474.80 μs |     82,857.369 μs |   4,541.692 μs |  1.31 |    0.18 |       No |
|     addition_RawTensor_Reference |     addition |       2048 |   float32 |       cuda |      6,318.71 μs |        335.180 μs |      18.372 μs |  0.25 |    0.00 |       No |
|        addition_Tensor_Reference |     addition |       2048 |   float32 |       cuda |      8,580.55 μs |      1,959.830 μs |     107.425 μs |  0.34 |    0.01 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                addScalar_PyTorch |    addScalar |       2048 |   float32 |       cuda |     33,549.74 μs |        955.305 μs |      52.363 μs |  1.00 |    0.00 |      Yes |
|             addScalar_TorchSharp |    addScalar |       2048 |   float32 |       cuda |     23,233.37 μs |        902.372 μs |      49.462 μs |  0.69 |    0.00 |       No |
|        addScalar_RawTensor_Torch |    addScalar |       2048 |   float32 |       cuda |    137,429.87 μs |     24,053.061 μs |   1,318.429 μs |  4.10 |    0.05 |       No |
|           addScalar_Tensor_Torch |    addScalar |       2048 |   float32 |       cuda |    164,763.63 μs |    560,904.479 μs |  30,745.067 μs |  4.91 |    0.92 |       No |
|    addScalar_RawTensor_Reference |    addScalar |       2048 |   float32 |       cuda |      2,640.96 μs |         58.524 μs |       3.208 μs |  0.08 |    0.00 |       No |
|       addScalar_Tensor_Reference |    addScalar |       2048 |   float32 |       cuda |      4,111.52 μs |      1,641.096 μs |      89.954 μs |  0.12 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|             addWithAlpha_PyTorch | addWithAlpha |       2048 |   float32 |       cuda |     14,544.49 μs |        909.568 μs |      49.856 μs |  1.00 |    0.00 |      Yes |
|          addWithAlpha_TorchSharp | addWithAlpha |       2048 |   float32 |       cuda |     18,681.60 μs |     31,382.246 μs |   1,720.167 μs |  1.28 |    0.11 |       No |
|     addWithAlpha_RawTensor_Torch | addWithAlpha |       2048 |   float32 |       cuda |     46,389.50 μs |      3,869.764 μs |     212.115 μs |  3.19 |    0.01 |       No |
|        addWithAlpha_Tensor_Torch | addWithAlpha |       2048 |   float32 |       cuda |     67,006.33 μs |     65,233.894 μs |   3,575.690 μs |  4.61 |    0.23 |       No |
| addWithAlpha_RawTensor_Reference | addWithAlpha |       2048 |   float32 |       cuda |      9,709.93 μs |      3,311.532 μs |     181.516 μs |  0.67 |    0.01 |       No |
|    addWithAlpha_Tensor_Reference | addWithAlpha |       2048 |   float32 |       cuda |     12,318.81 μs |      5,142.468 μs |     281.876 μs |  0.85 |    0.02 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|               addInPlace_PyTorch |   addInPlace |       2048 |   float32 |       cuda |     13,462.53 μs |      1,190.500 μs |      65.255 μs |  1.00 |    0.00 |      Yes |
|            addInPlace_TorchSharp |   addInPlace |       2048 |   float32 |       cuda |     12,204.57 μs |     28,592.936 μs |   1,567.275 μs |  0.91 |    0.12 |       No |
|       addInPlace_RawTensor_Torch |   addInPlace |       2048 |   float32 |       cuda |     21,236.60 μs |     18,746.835 μs |   1,027.577 μs |  1.58 |    0.07 |       No |
|          addInPlace_Tensor_Torch |   addInPlace |       2048 |   float32 |       cuda |     29,743.10 μs |     30,610.241 μs |   1,677.851 μs |  2.21 |    0.13 |       No |
|   addInPlace_RawTensor_Reference |   addInPlace |       2048 |   float32 |       cuda |      7,049.91 μs |      1,300.186 μs |      71.268 μs |  0.52 |    0.01 |       No |
|      addInPlace_Tensor_Reference |   addInPlace |       2048 |   float32 |       cuda |      7,917.59 μs |      3,166.866 μs |     173.587 μs |  0.59 |    0.02 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                   matmul_PyTorch |       matmul |       2048 |   float32 |       cuda |               NA |                NA |             NA |     ? |       ? |      Yes |
|                matmul_TorchSharp |       matmul |       2048 |   float32 |       cuda |      2,866.07 μs |      3,330.935 μs |     182.580 μs |     ? |       ? |       No |
|           matmul_RawTensor_Torch |       matmul |       2048 |   float32 |       cuda |      3,027.87 μs |      5,205.844 μs |     285.350 μs |     ? |       ? |       No |
|              matmul_Tensor_Torch |       matmul |       2048 |   float32 |       cuda |      3,807.00 μs |      4,010.548 μs |     219.832 μs |     ? |       ? |       No |
|       matmul_RawTensor_Reference |       matmul |       2048 |   float32 |       cuda |    206,034.90 μs |      7,659.018 μs |     419.817 μs |     ? |       ? |       No |
|          matmul_Tensor_Reference |       matmul |       2048 |   float32 |       cuda |    211,792.97 μs |     16,469.889 μs |     902.770 μs |     ? |       ? |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|              **fromCpuData_PyTorch** |  **fromCpuData** |       **2048** |   **float64** |        **cpu** |     **29,563.42 μs** |        **525.624 μs** |      **28.811 μs** | **1.000** |    **0.00** |      **Yes** |
|           fromCpuData_TorchSharp |  fromCpuData |       2048 |   float64 |        cpu |      1,561.17 μs |         30.485 μs |       1.671 μs | 0.053 |    0.00 |       No |
|      fromCpuData_RawTensor_Torch |  fromCpuData |       2048 |   float64 |        cpu |      4,330.42 μs |        363.751 μs |      19.938 μs | 0.146 |    0.00 |       No |
|         fromCpuData_Tensor_Torch |  fromCpuData |       2048 |   float64 |        cpu |      4,390.35 μs |        571.160 μs |      31.307 μs | 0.149 |    0.00 |       No |
|  fromCpuData_RawTensor_Reference |  fromCpuData |       2048 |   float64 |        cpu |        163.65 μs |         27.460 μs |       1.505 μs | 0.006 |    0.00 |       No |
|     fromCpuData_Tensor_Reference |  fromCpuData |       2048 |   float64 |        cpu |        180.03 μs |         47.050 μs |       2.579 μs | 0.006 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                    zeros_PyTorch |        zeros |       2048 |   float64 |        cpu |     18,495.38 μs |         89.384 μs |       4.899 μs |  1.00 |    0.00 |      Yes |
|                 zeros_TorchSharp |        zeros |       2048 |   float64 |        cpu |     11,524.45 μs |      1,086.677 μs |      59.564 μs |  0.62 |    0.00 |       No |
|            zeros_RawTensor_Torch |        zeros |       2048 |   float64 |        cpu |     20,387.38 μs |      3,001.464 μs |     164.520 μs |  1.10 |    0.01 |       No |
|               zeros_Tensor_Torch |        zeros |       2048 |   float64 |        cpu |     19,697.88 μs |      4,625.378 μs |     253.533 μs |  1.07 |    0.01 |       No |
|        zeros_RawTensor_Reference |        zeros |       2048 |   float64 |        cpu |      1,989.19 μs |        374.233 μs |      20.513 μs |  0.11 |    0.00 |       No |
|           zeros_Tensor_Reference |        zeros |       2048 |   float64 |        cpu |      1,982.68 μs |        368.189 μs |      20.182 μs |  0.11 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                     ones_PyTorch |         ones |       2048 |   float64 |        cpu |     16,469.69 μs |      1,585.537 μs |      86.909 μs |  1.00 |    0.00 |      Yes |
|                  ones_TorchSharp |         ones |       2048 |   float64 |        cpu |     11,423.92 μs |      2,216.395 μs |     121.488 μs |  0.69 |    0.01 |       No |
|             ones_RawTensor_Torch |         ones |       2048 |   float64 |        cpu |     19,334.17 μs |      1,993.394 μs |     109.265 μs |  1.17 |    0.01 |       No |
|                ones_Tensor_Torch |         ones |       2048 |   float64 |        cpu |     19,504.04 μs |      5,031.071 μs |     275.770 μs |  1.18 |    0.02 |       No |
|         ones_RawTensor_Reference |         ones |       2048 |   float64 |        cpu |      3,572.62 μs |      1,252.761 μs |      68.668 μs |  0.22 |    0.00 |       No |
|            ones_Tensor_Reference |         ones |       2048 |   float64 |        cpu |      3,567.07 μs |        952.982 μs |      52.236 μs |  0.22 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                     rand_PyTorch |         rand |       2048 |   float64 |        cpu |     56,473.59 μs |      2,580.300 μs |     141.435 μs |  1.00 |    0.00 |      Yes |
|                  rand_TorchSharp |         rand |       2048 |   float64 |        cpu |     27,876.04 μs |      3,168.312 μs |     173.666 μs |  0.49 |    0.00 |       No |
|             rand_RawTensor_Torch |         rand |       2048 |   float64 |        cpu |     54,426.19 μs |     25,093.739 μs |   1,375.473 μs |  0.96 |    0.03 |       No |
|                rand_Tensor_Torch |         rand |       2048 |   float64 |        cpu |     56,340.53 μs |     14,913.633 μs |     817.467 μs |  1.00 |    0.02 |       No |
|         rand_RawTensor_Reference |         rand |       2048 |   float64 |        cpu |     33,408.20 μs |      8,383.430 μs |     459.524 μs |  0.59 |    0.01 |       No |
|            rand_Tensor_Reference |         rand |       2048 |   float64 |        cpu |     34,509.99 μs |      1,397.823 μs |      76.619 μs |  0.61 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                 addition_PyTorch |     addition |       2048 |   float64 |        cpu |     12,514.28 μs |      1,387.075 μs |      76.030 μs |  1.00 |    0.00 |      Yes |
|              addition_TorchSharp |     addition |       2048 |   float64 |        cpu |     12,456.16 μs |      3,553.408 μs |     194.774 μs |  1.00 |    0.02 |       No |
|         addition_RawTensor_Torch |     addition |       2048 |   float64 |        cpu |     10,966.03 μs |      1,866.938 μs |     102.333 μs |  0.88 |    0.01 |       No |
|            addition_Tensor_Torch |     addition |       2048 |   float64 |        cpu |     15,008.51 μs |      4,729.433 μs |     259.236 μs |  1.20 |    0.03 |       No |
|     addition_RawTensor_Reference |     addition |       2048 |   float64 |        cpu |      7,135.58 μs |      2,022.049 μs |     110.835 μs |  0.57 |    0.01 |       No |
|        addition_Tensor_Reference |     addition |       2048 |   float64 |        cpu |      7,943.52 μs |      1,119.421 μs |      61.359 μs |  0.63 |    0.01 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                addScalar_PyTorch |    addScalar |       2048 |   float64 |        cpu |     23,503.14 μs |        670.881 μs |      36.773 μs |  1.00 |    0.00 |      Yes |
|             addScalar_TorchSharp |    addScalar |       2048 |   float64 |        cpu |     20,719.46 μs |        715.242 μs |      39.205 μs |  0.88 |    0.00 |       No |
|        addScalar_RawTensor_Torch |    addScalar |       2048 |   float64 |        cpu |     24,006.67 μs |      9,157.412 μs |     501.949 μs |  1.02 |    0.02 |       No |
|           addScalar_Tensor_Torch |    addScalar |       2048 |   float64 |        cpu |     29,814.23 μs |      2,643.498 μs |     144.899 μs |  1.27 |    0.00 |       No |
|    addScalar_RawTensor_Reference |    addScalar |       2048 |   float64 |        cpu |      2,706.03 μs |        757.906 μs |      41.543 μs |  0.12 |    0.00 |       No |
|       addScalar_Tensor_Reference |    addScalar |       2048 |   float64 |        cpu |      4,131.95 μs |      1,262.989 μs |      69.229 μs |  0.18 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|             addWithAlpha_PyTorch | addWithAlpha |       2048 |   float64 |        cpu |      8,681.34 μs |        976.963 μs |      53.551 μs |  1.00 |    0.00 |      Yes |
|          addWithAlpha_TorchSharp | addWithAlpha |       2048 |   float64 |        cpu |     11,612.75 μs |      2,260.447 μs |     123.903 μs |  1.34 |    0.01 |       No |
|     addWithAlpha_RawTensor_Torch | addWithAlpha |       2048 |   float64 |        cpu |     25,113.62 μs |        340.390 μs |      18.658 μs |  2.89 |    0.02 |       No |
|        addWithAlpha_Tensor_Torch | addWithAlpha |       2048 |   float64 |        cpu |     30,769.12 μs |     11,186.969 μs |     613.196 μs |  3.54 |    0.05 |       No |
| addWithAlpha_RawTensor_Reference | addWithAlpha |       2048 |   float64 |        cpu |      9,850.97 μs |      5,626.461 μs |     308.405 μs |  1.13 |    0.04 |       No |
|    addWithAlpha_Tensor_Reference | addWithAlpha |       2048 |   float64 |        cpu |     12,155.58 μs |      4,515.586 μs |     247.514 μs |  1.40 |    0.02 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|               addInPlace_PyTorch |   addInPlace |       2048 |   float64 |        cpu |      7,645.17 μs |         77.403 μs |       4.243 μs |  1.00 |    0.00 |      Yes |
|            addInPlace_TorchSharp |   addInPlace |       2048 |   float64 |        cpu |      4,790.29 μs |      1,371.312 μs |      75.166 μs |  0.63 |    0.01 |       No |
|       addInPlace_RawTensor_Torch |   addInPlace |       2048 |   float64 |        cpu |     11,182.70 μs |      1,319.234 μs |      72.312 μs |  1.46 |    0.01 |       No |
|          addInPlace_Tensor_Torch |   addInPlace |       2048 |   float64 |        cpu |     14,740.94 μs |      1,946.863 μs |     106.714 μs |  1.93 |    0.01 |       No |
|   addInPlace_RawTensor_Reference |   addInPlace |       2048 |   float64 |        cpu |      7,085.78 μs |      1,189.157 μs |      65.182 μs |  0.93 |    0.01 |       No |
|      addInPlace_Tensor_Reference |   addInPlace |       2048 |   float64 |        cpu |      7,892.52 μs |      1,287.652 μs |      70.581 μs |  1.03 |    0.01 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                   matmul_PyTorch |       matmul |       2048 |   float64 |        cpu |               NA |                NA |             NA |     ? |       ? |      Yes |
|                matmul_TorchSharp |       matmul |       2048 |   float64 |        cpu |      3,470.98 μs |      2,597.821 μs |     142.395 μs |     ? |       ? |       No |
|           matmul_RawTensor_Torch |       matmul |       2048 |   float64 |        cpu |      3,328.54 μs |      2,065.705 μs |     113.228 μs |     ? |       ? |       No |
|              matmul_Tensor_Torch |       matmul |       2048 |   float64 |        cpu |      4,062.56 μs |      1,901.304 μs |     104.217 μs |     ? |       ? |       No |
|       matmul_RawTensor_Reference |       matmul |       2048 |   float64 |        cpu |    210,219.23 μs |     87,903.107 μs |   4,818.266 μs |     ? |       ? |       No |
|          matmul_Tensor_Reference |       matmul |       2048 |   float64 |        cpu |    211,806.17 μs |     37,133.880 μs |   2,035.433 μs |     ? |       ? |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|              **fromCpuData_PyTorch** |  **fromCpuData** |       **2048** |   **float64** |       **cuda** |     **57,534.43 μs** |      **1,850.954 μs** |     **101.457 μs** | **1.000** |    **0.00** |      **Yes** |
|           fromCpuData_TorchSharp |  fromCpuData |       2048 |   float64 |       cuda |      1,168.73 μs |        971.871 μs |      53.272 μs | 0.020 |    0.00 |       No |
|      fromCpuData_RawTensor_Torch |  fromCpuData |       2048 |   float64 |       cuda |     26,517.10 μs |     45,645.611 μs |   2,501.990 μs | 0.461 |    0.04 |       No |
|         fromCpuData_Tensor_Torch |  fromCpuData |       2048 |   float64 |       cuda |     25,931.53 μs |     11,571.316 μs |     634.263 μs | 0.451 |    0.01 |       No |
|  fromCpuData_RawTensor_Reference |  fromCpuData |       2048 |   float64 |       cuda |        160.03 μs |         70.692 μs |       3.875 μs | 0.003 |    0.00 |       No |
|     fromCpuData_Tensor_Reference |  fromCpuData |       2048 |   float64 |       cuda |        179.56 μs |         53.311 μs |       2.922 μs | 0.003 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                    zeros_PyTorch |        zeros |       2048 |   float64 |       cuda |     39,555.68 μs |      2,899.598 μs |     158.937 μs |  1.00 |    0.00 |      Yes |
|                 zeros_TorchSharp |        zeros |       2048 |   float64 |       cuda |     21,256.33 μs |      7,857.958 μs |     430.721 μs |  0.54 |    0.01 |       No |
|            zeros_RawTensor_Torch |        zeros |       2048 |   float64 |       cuda |     22,345.53 μs |      2,828.009 μs |     155.013 μs |  0.56 |    0.01 |       No |
|               zeros_Tensor_Torch |        zeros |       2048 |   float64 |       cuda |     24,683.13 μs |     31,310.460 μs |   1,716.232 μs |  0.62 |    0.05 |       No |
|        zeros_RawTensor_Reference |        zeros |       2048 |   float64 |       cuda |      1,984.31 μs |        306.487 μs |      16.800 μs |  0.05 |    0.00 |       No |
|           zeros_Tensor_Reference |        zeros |       2048 |   float64 |       cuda |      2,033.80 μs |        659.950 μs |      36.174 μs |  0.05 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                     ones_PyTorch |         ones |       2048 |   float64 |       cuda |     39,589.27 μs |      2,650.776 μs |     145.298 μs |  1.00 |    0.00 |      Yes |
|                  ones_TorchSharp |         ones |       2048 |   float64 |       cuda |     23,305.47 μs |      2,487.958 μs |     136.373 μs |  0.59 |    0.01 |       No |
|             ones_RawTensor_Torch |         ones |       2048 |   float64 |       cuda |     22,631.67 μs |      8,556.538 μs |     469.013 μs |  0.57 |    0.01 |       No |
|                ones_Tensor_Torch |         ones |       2048 |   float64 |       cuda |     25,312.43 μs |      2,337.072 μs |     128.103 μs |  0.64 |    0.01 |       No |
|         ones_RawTensor_Reference |         ones |       2048 |   float64 |       cuda |      3,559.03 μs |        578.885 μs |      31.731 μs |  0.09 |    0.00 |       No |
|            ones_Tensor_Reference |         ones |       2048 |   float64 |       cuda |      3,556.99 μs |        849.653 μs |      46.572 μs |  0.09 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                     rand_PyTorch |         rand |       2048 |   float64 |       cuda |     42,589.52 μs |      1,576.280 μs |      86.401 μs |  1.00 |    0.00 |      Yes |
|                  rand_TorchSharp |         rand |       2048 |   float64 |       cuda |     22,085.13 μs |      1,419.358 μs |      77.800 μs |  0.52 |    0.00 |       No |
|             rand_RawTensor_Torch |         rand |       2048 |   float64 |       cuda |     24,249.33 μs |      8,799.184 μs |     482.313 μs |  0.57 |    0.01 |       No |
|                rand_Tensor_Torch |         rand |       2048 |   float64 |       cuda |     26,865.77 μs |      6,071.432 μs |     332.796 μs |  0.63 |    0.01 |       No |
|         rand_RawTensor_Reference |         rand |       2048 |   float64 |       cuda |     35,216.13 μs |      2,422.445 μs |     132.782 μs |  0.83 |    0.00 |       No |
|            rand_Tensor_Reference |         rand |       2048 |   float64 |       cuda |     36,578.64 μs |     17,808.800 μs |     976.160 μs |  0.86 |    0.02 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                 addition_PyTorch |     addition |       2048 |   float64 |       cuda |     24,494.58 μs |        395.315 μs |      21.669 μs |  1.00 |    0.00 |      Yes |
|              addition_TorchSharp |     addition |       2048 |   float64 |       cuda |     21,420.10 μs |      2,444.158 μs |     133.973 μs |  0.87 |    0.01 |       No |
|         addition_RawTensor_Torch |     addition |       2048 |   float64 |       cuda |     20,059.80 μs |      7,501.606 μs |     411.188 μs |  0.82 |    0.02 |       No |
|            addition_Tensor_Torch |     addition |       2048 |   float64 |       cuda |     33,074.03 μs |      7,959.618 μs |     436.294 μs |  1.35 |    0.02 |       No |
|     addition_RawTensor_Reference |     addition |       2048 |   float64 |       cuda |      6,397.62 μs |        498.389 μs |      27.318 μs |  0.26 |    0.00 |       No |
|        addition_Tensor_Reference |     addition |       2048 |   float64 |       cuda |      8,499.59 μs |        359.349 μs |      19.697 μs |  0.35 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                addScalar_PyTorch |    addScalar |       2048 |   float64 |       cuda |     32,628.92 μs |        920.793 μs |      50.472 μs |  1.00 |    0.00 |      Yes |
|             addScalar_TorchSharp |    addScalar |       2048 |   float64 |       cuda |     24,295.37 μs |      3,851.758 μs |     211.128 μs |  0.74 |    0.01 |       No |
|        addScalar_RawTensor_Torch |    addScalar |       2048 |   float64 |       cuda |    168,464.13 μs |    263,929.439 μs |  14,466.863 μs |  5.16 |    0.44 |       No |
|           addScalar_Tensor_Torch |    addScalar |       2048 |   float64 |       cuda |    146,867.70 μs |      9,861.718 μs |     540.554 μs |  4.50 |    0.02 |       No |
|    addScalar_RawTensor_Reference |    addScalar |       2048 |   float64 |       cuda |      2,651.09 μs |        757.227 μs |      41.506 μs |  0.08 |    0.00 |       No |
|       addScalar_Tensor_Reference |    addScalar |       2048 |   float64 |       cuda |      4,173.31 μs |         73.876 μs |       4.049 μs |  0.13 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|             addWithAlpha_PyTorch | addWithAlpha |       2048 |   float64 |       cuda |     14,526.91 μs |        794.492 μs |      43.549 μs |  1.00 |    0.00 |      Yes |
|          addWithAlpha_TorchSharp | addWithAlpha |       2048 |   float64 |       cuda |     17,551.93 μs |      3,128.996 μs |     171.511 μs |  1.21 |    0.01 |       No |
|     addWithAlpha_RawTensor_Torch | addWithAlpha |       2048 |   float64 |       cuda |     47,750.40 μs |     46,637.801 μs |   2,556.375 μs |  3.29 |    0.19 |       No |
|        addWithAlpha_Tensor_Torch | addWithAlpha |       2048 |   float64 |       cuda |     66,359.50 μs |     99,011.427 μs |   5,427.150 μs |  4.57 |    0.36 |       No |
| addWithAlpha_RawTensor_Reference | addWithAlpha |       2048 |   float64 |       cuda |      9,765.92 μs |      2,569.452 μs |     140.840 μs |  0.67 |    0.01 |       No |
|    addWithAlpha_Tensor_Reference | addWithAlpha |       2048 |   float64 |       cuda |     11,720.74 μs |      5,319.435 μs |     291.576 μs |  0.81 |    0.02 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|               addInPlace_PyTorch |   addInPlace |       2048 |   float64 |       cuda |     13,561.30 μs |        764.445 μs |      41.902 μs |  1.00 |    0.00 |      Yes |
|            addInPlace_TorchSharp |   addInPlace |       2048 |   float64 |       cuda |     11,662.90 μs |     30,556.530 μs |   1,674.907 μs |  0.86 |    0.12 |       No |
|       addInPlace_RawTensor_Torch |   addInPlace |       2048 |   float64 |       cuda |     19,875.90 μs |      7,644.542 μs |     419.023 μs |  1.47 |    0.04 |       No |
|          addInPlace_Tensor_Torch |   addInPlace |       2048 |   float64 |       cuda |     29,671.83 μs |      3,486.329 μs |     191.097 μs |  2.19 |    0.02 |       No |
|   addInPlace_RawTensor_Reference |   addInPlace |       2048 |   float64 |       cuda |      6,370.84 μs |        870.742 μs |      47.728 μs |  0.47 |    0.00 |       No |
|      addInPlace_Tensor_Reference |   addInPlace |       2048 |   float64 |       cuda |      8,613.02 μs |      1,508.931 μs |      82.710 μs |  0.64 |    0.01 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                   matmul_PyTorch |       matmul |       2048 |   float64 |       cuda |               NA |                NA |             NA |     ? |       ? |      Yes |
|                matmul_TorchSharp |       matmul |       2048 |   float64 |       cuda |      2,288.13 μs |      3,327.503 μs |     182.392 μs |     ? |       ? |       No |
|           matmul_RawTensor_Torch |       matmul |       2048 |   float64 |       cuda |      2,309.87 μs |        345.488 μs |      18.937 μs |     ? |       ? |       No |
|              matmul_Tensor_Torch |       matmul |       2048 |   float64 |       cuda |      3,128.17 μs |      3,571.375 μs |     195.759 μs |     ? |       ? |       No |
|       matmul_RawTensor_Reference |       matmul |       2048 |   float64 |       cuda |    209,704.97 μs |     78,279.985 μs |   4,290.790 μs |     ? |       ? |       No |
|          matmul_Tensor_Reference |       matmul |       2048 |   float64 |       cuda |    211,331.47 μs |     21,429.152 μs |   1,174.604 μs |     ? |       ? |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|              **fromCpuData_PyTorch** |  **fromCpuData** |       **2048** |     **int32** |        **cpu** |     **22,549.19 μs** |      **1,318.081 μs** |      **72.248 μs** | **1.000** |    **0.00** |      **Yes** |
|           fromCpuData_TorchSharp |  fromCpuData |       2048 |     int32 |        cpu |      1,625.27 μs |        419.322 μs |      22.984 μs | 0.072 |    0.00 |       No |
|      fromCpuData_RawTensor_Torch |  fromCpuData |       2048 |     int32 |        cpu |      3,808.10 μs |        414.122 μs |      22.699 μs | 0.169 |    0.00 |       No |
|         fromCpuData_Tensor_Torch |  fromCpuData |       2048 |     int32 |        cpu |      3,793.48 μs |        715.944 μs |      39.243 μs | 0.168 |    0.00 |       No |
|  fromCpuData_RawTensor_Reference |  fromCpuData |       2048 |     int32 |        cpu |        112.55 μs |          9.760 μs |       0.535 μs | 0.005 |    0.00 |       No |
|     fromCpuData_Tensor_Reference |  fromCpuData |       2048 |     int32 |        cpu |        119.42 μs |         44.634 μs |       2.447 μs | 0.005 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                    zeros_PyTorch |        zeros |       2048 |     int32 |        cpu |     14,544.67 μs |        734.600 μs |      40.266 μs |  1.00 |    0.00 |      Yes |
|                 zeros_TorchSharp |        zeros |       2048 |     int32 |        cpu |     11,991.24 μs |      2,077.322 μs |     113.865 μs |  0.82 |    0.01 |       No |
|            zeros_RawTensor_Torch |        zeros |       2048 |     int32 |        cpu |     10,091.44 μs |      2,398.786 μs |     131.486 μs |  0.69 |    0.01 |       No |
|               zeros_Tensor_Torch |        zeros |       2048 |     int32 |        cpu |     10,498.69 μs |      3,629.544 μs |     198.948 μs |  0.72 |    0.01 |       No |
|        zeros_RawTensor_Reference |        zeros |       2048 |     int32 |        cpu |      1,160.12 μs |        180.201 μs |       9.877 μs |  0.08 |    0.00 |       No |
|           zeros_Tensor_Reference |        zeros |       2048 |     int32 |        cpu |      1,160.73 μs |        678.936 μs |      37.215 μs |  0.08 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                     ones_PyTorch |         ones |       2048 |     int32 |        cpu |     14,532.54 μs |        326.730 μs |      17.909 μs |  1.00 |    0.00 |      Yes |
|                  ones_TorchSharp |         ones |       2048 |     int32 |        cpu |     11,261.97 μs |      1,195.869 μs |      65.550 μs |  0.77 |    0.01 |       No |
|             ones_RawTensor_Torch |         ones |       2048 |     int32 |        cpu |     10,076.63 μs |      3,995.906 μs |     219.029 μs |  0.69 |    0.01 |       No |
|                ones_Tensor_Torch |         ones |       2048 |     int32 |        cpu |     10,629.00 μs |      3,451.427 μs |     189.184 μs |  0.73 |    0.01 |       No |
|         ones_RawTensor_Reference |         ones |       2048 |     int32 |        cpu |      2,637.85 μs |        560.191 μs |      30.706 μs |  0.18 |    0.00 |       No |
|            ones_Tensor_Reference |         ones |       2048 |     int32 |        cpu |      2,680.24 μs |        335.366 μs |      18.383 μs |  0.18 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                     rand_PyTorch |         rand |       2048 |     int32 |        cpu |               NA |                NA |             NA |     ? |       ? |      Yes |
|                  rand_TorchSharp |         rand |       2048 |     int32 |        cpu |     40,562.35 μs |      2,417.750 μs |     132.525 μs |     ? |       ? |       No |
|             rand_RawTensor_Torch |         rand |       2048 |     int32 |        cpu |               NA |                NA |             NA |     ? |       ? |       No |
|                rand_Tensor_Torch |         rand |       2048 |     int32 |        cpu |               NA |                NA |             NA |     ? |       ? |       No |
|         rand_RawTensor_Reference |         rand |       2048 |     int32 |        cpu |               NA |                NA |             NA |     ? |       ? |       No |
|            rand_Tensor_Reference |         rand |       2048 |     int32 |        cpu |               NA |                NA |             NA |     ? |       ? |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                 addition_PyTorch |     addition |       2048 |     int32 |        cpu |      9,591.35 μs |      1,238.951 μs |      67.911 μs |  1.00 |    0.00 |      Yes |
|              addition_TorchSharp |     addition |       2048 |     int32 |        cpu |     11,998.58 μs |        483.946 μs |      26.527 μs |  1.25 |    0.01 |       No |
|         addition_RawTensor_Torch |     addition |       2048 |     int32 |        cpu |     11,157.03 μs |      2,289.993 μs |     125.522 μs |  1.16 |    0.02 |       No |
|            addition_Tensor_Torch |     addition |       2048 |     int32 |        cpu |     13,962.50 μs |      1,738.660 μs |      95.302 μs |  1.46 |    0.02 |       No |
|     addition_RawTensor_Reference |     addition |       2048 |     int32 |        cpu |      6,364.65 μs |      1,636.636 μs |      89.710 μs |  0.66 |    0.01 |       No |
|        addition_Tensor_Reference |     addition |       2048 |     int32 |        cpu |      7,922.91 μs |        976.891 μs |      53.547 μs |  0.83 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                addScalar_PyTorch |    addScalar |       2048 |     int32 |        cpu |     18,503.70 μs |      1,905.782 μs |     104.462 μs |  1.00 |    0.00 |      Yes |
|             addScalar_TorchSharp |    addScalar |       2048 |     int32 |        cpu |     21,363.00 μs |      5,574.660 μs |     305.566 μs |  1.15 |    0.02 |       No |
|        addScalar_RawTensor_Torch |    addScalar |       2048 |     int32 |        cpu |     29,373.25 μs |      2,470.534 μs |     135.418 μs |  1.59 |    0.01 |       No |
|           addScalar_Tensor_Torch |    addScalar |       2048 |     int32 |        cpu |     44,347.01 μs |     26,775.287 μs |   1,467.644 μs |  2.40 |    0.09 |       No |
|    addScalar_RawTensor_Reference |    addScalar |       2048 |     int32 |        cpu |               NA |                NA |             NA |     ? |       ? |       No |
|       addScalar_Tensor_Reference |    addScalar |       2048 |     int32 |        cpu |    135,940.00 μs |      7,814.580 μs |     428.344 μs |  7.35 |    0.02 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|             addWithAlpha_PyTorch | addWithAlpha |       2048 |     int32 |        cpu |      6,623.87 μs |        691.562 μs |      37.907 μs |  1.00 |    0.00 |      Yes |
|          addWithAlpha_TorchSharp | addWithAlpha |       2048 |     int32 |        cpu |     12,095.67 μs |      1,731.840 μs |      94.928 μs |  1.83 |    0.02 |       No |
|     addWithAlpha_RawTensor_Torch | addWithAlpha |       2048 |     int32 |        cpu |     41,775.94 μs |     11,950.107 μs |     655.026 μs |  6.31 |    0.11 |       No |
|        addWithAlpha_Tensor_Torch | addWithAlpha |       2048 |     int32 |        cpu |     68,029.77 μs |     10,324.501 μs |     565.921 μs | 10.27 |    0.11 |       No |
| addWithAlpha_RawTensor_Reference | addWithAlpha |       2048 |     int32 |        cpu |               NA |                NA |             NA |     ? |       ? |       No |
|    addWithAlpha_Tensor_Reference | addWithAlpha |       2048 |     int32 |        cpu |    276,343.30 μs |     41,424.862 μs |   2,270.636 μs | 41.72 |    0.58 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|               addInPlace_PyTorch |   addInPlace |       2048 |     int32 |        cpu |      5,673.69 μs |        609.165 μs |      33.390 μs |  1.00 |    0.00 |      Yes |
|            addInPlace_TorchSharp |   addInPlace |       2048 |     int32 |        cpu |      4,600.91 μs |      1,615.493 μs |      88.551 μs |  0.81 |    0.01 |       No |
|       addInPlace_RawTensor_Torch |   addInPlace |       2048 |     int32 |        cpu |     11,095.06 μs |      2,354.659 μs |     129.067 μs |  1.96 |    0.01 |       No |
|          addInPlace_Tensor_Torch |   addInPlace |       2048 |     int32 |        cpu |     14,717.35 μs |      8,278.504 μs |     453.773 μs |  2.59 |    0.09 |       No |
|   addInPlace_RawTensor_Reference |   addInPlace |       2048 |     int32 |        cpu |      6,332.17 μs |      1,443.269 μs |      79.110 μs |  1.12 |    0.01 |       No |
|      addInPlace_Tensor_Reference |   addInPlace |       2048 |     int32 |        cpu |      8,602.59 μs |        973.614 μs |      53.367 μs |  1.52 |    0.01 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                   matmul_PyTorch |       matmul |       2048 |     int32 |        cpu |               NA |                NA |             NA |     ? |       ? |      Yes |
|                matmul_TorchSharp |       matmul |       2048 |     int32 |        cpu |      6,905.56 μs |        213.843 μs |      11.721 μs |     ? |       ? |       No |
|           matmul_RawTensor_Torch |       matmul |       2048 |     int32 |        cpu |      6,780.89 μs |      1,740.906 μs |      95.425 μs |     ? |       ? |       No |
|              matmul_Tensor_Torch |       matmul |       2048 |     int32 |        cpu |      7,099.89 μs |      1,148.401 μs |      62.948 μs |     ? |       ? |       No |
|       matmul_RawTensor_Reference |       matmul |       2048 |     int32 |        cpu |    209,865.97 μs |     30,116.344 μs |   1,650.778 μs |     ? |       ? |       No |
|          matmul_Tensor_Reference |       matmul |       2048 |     int32 |        cpu |    219,412.17 μs |    181,000.181 μs |   9,921.231 μs |     ? |       ? |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|              **fromCpuData_PyTorch** |  **fromCpuData** |       **2048** |     **int32** |       **cuda** |     **45,562.05 μs** |      **2,101.523 μs** |     **115.192 μs** | **1.000** |    **0.00** |      **Yes** |
|           fromCpuData_TorchSharp |  fromCpuData |       2048 |     int32 |       cuda |      1,113.37 μs |        404.107 μs |      22.150 μs | 0.024 |    0.00 |       No |
|      fromCpuData_RawTensor_Torch |  fromCpuData |       2048 |     int32 |       cuda |     25,038.83 μs |     13,398.859 μs |     734.437 μs | 0.550 |    0.02 |       No |
|         fromCpuData_Tensor_Torch |  fromCpuData |       2048 |     int32 |       cuda |     26,512.03 μs |     17,336.235 μs |     950.258 μs | 0.582 |    0.02 |       No |
|  fromCpuData_RawTensor_Reference |  fromCpuData |       2048 |     int32 |       cuda |        114.62 μs |         16.893 μs |       0.926 μs | 0.003 |    0.00 |       No |
|     fromCpuData_Tensor_Reference |  fromCpuData |       2048 |     int32 |       cuda |        120.72 μs |         44.148 μs |       2.420 μs | 0.003 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                    zeros_PyTorch |        zeros |       2048 |     int32 |       cuda |     37,567.05 μs |        781.707 μs |      42.848 μs |  1.00 |    0.00 |      Yes |
|                 zeros_TorchSharp |        zeros |       2048 |     int32 |       cuda |     20,512.33 μs |      2,993.515 μs |     164.085 μs |  0.55 |    0.00 |       No |
|            zeros_RawTensor_Torch |        zeros |       2048 |     int32 |       cuda |     21,892.80 μs |      5,211.297 μs |     285.649 μs |  0.58 |    0.01 |       No |
|               zeros_Tensor_Torch |        zeros |       2048 |     int32 |       cuda |     22,617.20 μs |      5,073.907 μs |     278.118 μs |  0.60 |    0.01 |       No |
|        zeros_RawTensor_Reference |        zeros |       2048 |     int32 |       cuda |      1,142.32 μs |        477.911 μs |      26.196 μs |  0.03 |    0.00 |       No |
|           zeros_Tensor_Reference |        zeros |       2048 |     int32 |       cuda |      1,198.41 μs |        971.434 μs |      53.248 μs |  0.03 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                     ones_PyTorch |         ones |       2048 |     int32 |       cuda |     40,493.31 μs |      1,180.620 μs |      64.714 μs |  1.00 |    0.00 |      Yes |
|                  ones_TorchSharp |         ones |       2048 |     int32 |       cuda |     20,159.30 μs |      2,157.663 μs |     118.269 μs |  0.50 |    0.00 |       No |
|             ones_RawTensor_Torch |         ones |       2048 |     int32 |       cuda |     22,022.00 μs |      4,702.019 μs |     257.734 μs |  0.54 |    0.01 |       No |
|                ones_Tensor_Torch |         ones |       2048 |     int32 |       cuda |     28,603.30 μs |      2,955.312 μs |     161.991 μs |  0.71 |    0.00 |       No |
|         ones_RawTensor_Reference |         ones |       2048 |     int32 |       cuda |      2,664.92 μs |      1,116.445 μs |      61.196 μs |  0.07 |    0.00 |       No |
|            ones_Tensor_Reference |         ones |       2048 |     int32 |       cuda |      2,726.54 μs |      1,373.510 μs |      75.287 μs |  0.07 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                     rand_PyTorch |         rand |       2048 |     int32 |       cuda |               NA |                NA |             NA |     ? |       ? |      Yes |
|                  rand_TorchSharp |         rand |       2048 |     int32 |       cuda |     22,212.13 μs |      5,433.588 μs |     297.833 μs |     ? |       ? |       No |
|             rand_RawTensor_Torch |         rand |       2048 |     int32 |       cuda |               NA |                NA |             NA |     ? |       ? |       No |
|                rand_Tensor_Torch |         rand |       2048 |     int32 |       cuda |               NA |                NA |             NA |     ? |       ? |       No |
|         rand_RawTensor_Reference |         rand |       2048 |     int32 |       cuda |               NA |                NA |             NA |     ? |       ? |       No |
|            rand_Tensor_Reference |         rand |       2048 |     int32 |       cuda |               NA |                NA |             NA |     ? |       ? |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                 addition_PyTorch |     addition |       2048 |     int32 |       cuda |     25,633.62 μs |      2,335.875 μs |     128.037 μs |  1.00 |    0.00 |      Yes |
|              addition_TorchSharp |     addition |       2048 |     int32 |       cuda |     18,492.53 μs |      9,638.583 μs |     528.323 μs |  0.72 |    0.02 |       No |
|         addition_RawTensor_Torch |     addition |       2048 |     int32 |       cuda |     20,276.23 μs |      3,296.713 μs |     180.704 μs |  0.79 |    0.01 |       No |
|            addition_Tensor_Torch |     addition |       2048 |     int32 |       cuda |     29,667.23 μs |     16,838.870 μs |     922.995 μs |  1.16 |    0.04 |       No |
|     addition_RawTensor_Reference |     addition |       2048 |     int32 |       cuda |      6,310.39 μs |        682.542 μs |      37.412 μs |  0.25 |    0.00 |       No |
|        addition_Tensor_Reference |     addition |       2048 |     int32 |       cuda |      8,715.64 μs |      2,180.653 μs |     119.529 μs |  0.34 |    0.01 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                addScalar_PyTorch |    addScalar |       2048 |     int32 |       cuda |     32,654.60 μs |      1,485.904 μs |      81.447 μs |  1.00 |    0.00 |      Yes |
|             addScalar_TorchSharp |    addScalar |       2048 |     int32 |       cuda |     22,763.87 μs |      1,893.653 μs |     103.798 μs |  0.70 |    0.00 |       No |
|        addScalar_RawTensor_Torch |    addScalar |       2048 |     int32 |       cuda |    135,524.93 μs |     12,007.926 μs |     658.195 μs |  4.15 |    0.01 |       No |
|           addScalar_Tensor_Torch |    addScalar |       2048 |     int32 |       cuda |    194,791.27 μs |     83,952.075 μs |   4,601.697 μs |  5.97 |    0.13 |       No |
|    addScalar_RawTensor_Reference |    addScalar |       2048 |     int32 |       cuda |               NA |                NA |             NA |     ? |       ? |       No |
|       addScalar_Tensor_Reference |    addScalar |       2048 |     int32 |       cuda |    131,643.23 μs |     21,679.652 μs |   1,188.335 μs |  4.03 |    0.03 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|             addWithAlpha_PyTorch | addWithAlpha |       2048 |     int32 |       cuda |     17,566.04 μs |        847.107 μs |      46.433 μs |  1.00 |    0.00 |      Yes |
|          addWithAlpha_TorchSharp | addWithAlpha |       2048 |     int32 |       cuda |     18,497.50 μs |      2,779.805 μs |     152.371 μs |  1.05 |    0.01 |       No |
|     addWithAlpha_RawTensor_Torch | addWithAlpha |       2048 |     int32 |       cuda |     49,423.17 μs |      4,918.849 μs |     269.619 μs |  2.81 |    0.02 |       No |
|        addWithAlpha_Tensor_Torch | addWithAlpha |       2048 |     int32 |       cuda |    140,869.93 μs |    195,788.609 μs |  10,731.834 μs |  8.02 |    0.61 |       No |
| addWithAlpha_RawTensor_Reference | addWithAlpha |       2048 |     int32 |       cuda |               NA |                NA |             NA |     ? |       ? |       No |
|    addWithAlpha_Tensor_Reference | addWithAlpha |       2048 |     int32 |       cuda |    274,835.53 μs |    140,609.330 μs |   7,707.272 μs | 15.65 |    0.46 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|               addInPlace_PyTorch |   addInPlace |       2048 |     int32 |       cuda |     12,605.23 μs |      1,193.846 μs |      65.439 μs |  1.00 |    0.00 |      Yes |
|            addInPlace_TorchSharp |   addInPlace |       2048 |     int32 |       cuda |     10,147.57 μs |        282.715 μs |      15.497 μs |  0.81 |    0.00 |       No |
|       addInPlace_RawTensor_Torch |   addInPlace |       2048 |     int32 |       cuda |     19,622.17 μs |      7,643.038 μs |     418.941 μs |  1.56 |    0.03 |       No |
|          addInPlace_Tensor_Torch |   addInPlace |       2048 |     int32 |       cuda |     28,750.67 μs |      6,609.705 μs |     362.300 μs |  2.28 |    0.02 |       No |
|   addInPlace_RawTensor_Reference |   addInPlace |       2048 |     int32 |       cuda |      7,022.96 μs |      1,129.914 μs |      61.934 μs |  0.56 |    0.01 |       No |
|      addInPlace_Tensor_Reference |   addInPlace |       2048 |     int32 |       cuda |      8,575.25 μs |      1,615.524 μs |      88.552 μs |  0.68 |    0.01 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                   matmul_PyTorch |       matmul |       2048 |     int32 |       cuda |               NA |                NA |             NA |     ? |       ? |      Yes |
|                matmul_TorchSharp |       matmul |       2048 |     int32 |       cuda |               NA |                NA |             NA |     ? |       ? |       No |
|           matmul_RawTensor_Torch |       matmul |       2048 |     int32 |       cuda |     13,825.53 μs |      6,863.919 μs |     376.235 μs |     ? |       ? |       No |
|              matmul_Tensor_Torch |       matmul |       2048 |     int32 |       cuda |     14,929.63 μs |      5,541.999 μs |     303.776 μs |     ? |       ? |       No |
|       matmul_RawTensor_Reference |       matmul |       2048 |     int32 |       cuda |    214,213.63 μs |     41,391.752 μs |   2,268.822 μs |     ? |       ? |       No |
|          matmul_Tensor_Reference |       matmul |       2048 |     int32 |       cuda |    210,900.33 μs |     30,477.862 μs |   1,670.594 μs |     ? |       ? |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|              **fromCpuData_PyTorch** |  **fromCpuData** |      **65536** |   **float32** |        **cpu** |     **30,555.24 μs** |      **1,240.561 μs** |      **67.999 μs** | **1.000** |    **0.00** |      **Yes** |
|           fromCpuData_TorchSharp |  fromCpuData |      65536 |   float32 |        cpu |         42.53 μs |         27.491 μs |       1.507 μs | 0.001 |    0.00 |       No |
|      fromCpuData_RawTensor_Torch |  fromCpuData |      65536 |   float32 |        cpu |      1,710.01 μs |        180.618 μs |       9.900 μs | 0.056 |    0.00 |       No |
|         fromCpuData_Tensor_Torch |  fromCpuData |      65536 |   float32 |        cpu |      1,858.80 μs |        152.555 μs |       8.362 μs | 0.061 |    0.00 |       No |
|  fromCpuData_RawTensor_Reference |  fromCpuData |      65536 |   float32 |        cpu |      2,565.57 μs |      1,231.321 μs |      67.493 μs | 0.084 |    0.00 |       No |
|     fromCpuData_Tensor_Reference |  fromCpuData |      65536 |   float32 |        cpu |      2,658.28 μs |      1,885.422 μs |     103.346 μs | 0.087 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                    zeros_PyTorch |        zeros |      65536 |   float32 |        cpu |      1,865.15 μs |        328.858 μs |      18.026 μs |  1.00 |    0.00 |      Yes |
|                 zeros_TorchSharp |        zeros |      65536 |   float32 |        cpu |      8,657.19 μs |      2,931.423 μs |     160.681 μs |  4.64 |    0.04 |       No |
|            zeros_RawTensor_Torch |        zeros |      65536 |   float32 |        cpu |      4,882.50 μs |      1,661.149 μs |      91.053 μs |  2.62 |    0.04 |       No |
|               zeros_Tensor_Torch |        zeros |      65536 |   float32 |        cpu |      4,850.83 μs |      2,380.717 μs |     130.495 μs |  2.60 |    0.09 |       No |
|        zeros_RawTensor_Reference |        zeros |      65536 |   float32 |        cpu |      3,668.72 μs |      1,159.099 μs |      63.534 μs |  1.97 |    0.02 |       No |
|           zeros_Tensor_Reference |        zeros |      65536 |   float32 |        cpu |      3,804.43 μs |      1,895.340 μs |     103.890 μs |  2.04 |    0.05 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                     ones_PyTorch |         ones |      65536 |   float32 |        cpu |      1,867.53 μs |        264.018 μs |      14.472 μs |  1.00 |    0.00 |      Yes |
|                  ones_TorchSharp |         ones |      65536 |   float32 |        cpu |      8,609.49 μs |      3,049.774 μs |     167.168 μs |  4.61 |    0.06 |       No |
|             ones_RawTensor_Torch |         ones |      65536 |   float32 |        cpu |      4,812.56 μs |      2,202.417 μs |     120.722 μs |  2.58 |    0.06 |       No |
|                ones_Tensor_Torch |         ones |      65536 |   float32 |        cpu |      4,892.62 μs |      2,214.348 μs |     121.376 μs |  2.62 |    0.05 |       No |
|         ones_RawTensor_Reference |         ones |      65536 |   float32 |        cpu |      5,700.70 μs |      1,491.633 μs |      81.761 μs |  3.05 |    0.06 |       No |
|            ones_Tensor_Reference |         ones |      65536 |   float32 |        cpu |      5,674.43 μs |        781.562 μs |      42.840 μs |  3.04 |    0.04 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                     rand_PyTorch |         rand |      65536 |   float32 |        cpu |     18,572.29 μs |      1,488.901 μs |      81.612 μs |  1.00 |    0.00 |      Yes |
|                  rand_TorchSharp |         rand |      65536 |   float32 |        cpu |     43,877.32 μs |     12,174.947 μs |     667.350 μs |  2.36 |    0.05 |       No |
|             rand_RawTensor_Torch |         rand |      65536 |   float32 |        cpu |     20,685.95 μs |      2,275.807 μs |     124.745 μs |  1.11 |    0.01 |       No |
|                rand_Tensor_Torch |         rand |      65536 |   float32 |        cpu |     21,062.02 μs |      3,304.351 μs |     181.123 μs |  1.13 |    0.00 |       No |
|         rand_RawTensor_Reference |         rand |      65536 |   float32 |        cpu |     37,904.20 μs |      9,234.396 μs |     506.168 μs |  2.04 |    0.03 |       No |
|            rand_Tensor_Reference |         rand |      65536 |   float32 |        cpu |     35,858.50 μs |      3,278.763 μs |     179.720 μs |  1.93 |    0.01 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                 addition_PyTorch |     addition |      65536 |   float32 |        cpu |      7,652.43 μs |      1,291.117 μs |      70.770 μs |  1.00 |    0.00 |      Yes |
|              addition_TorchSharp |     addition |      65536 |   float32 |        cpu |      5,800.29 μs |        985.716 μs |      54.030 μs |  0.76 |    0.00 |       No |
|         addition_RawTensor_Torch |     addition |      65536 |   float32 |        cpu |      6,020.40 μs |      3,272.014 μs |     179.350 μs |  0.79 |    0.02 |       No |
|            addition_Tensor_Torch |     addition |      65536 |   float32 |        cpu |      6,327.07 μs |      1,665.367 μs |      91.284 μs |  0.83 |    0.01 |       No |
|     addition_RawTensor_Reference |     addition |      65536 |   float32 |        cpu |      9,297.57 μs |        837.954 μs |      45.931 μs |  1.22 |    0.01 |       No |
|        addition_Tensor_Reference |     addition |      65536 |   float32 |        cpu |      9,603.37 μs |      5,507.782 μs |     301.900 μs |  1.26 |    0.04 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                addScalar_PyTorch |    addScalar |      65536 |   float32 |        cpu |      7,769.25 μs |      2,794.743 μs |     153.189 μs |  1.00 |    0.00 |      Yes |
|             addScalar_TorchSharp |    addScalar |      65536 |   float32 |        cpu |      6,298.97 μs |        971.686 μs |      53.261 μs |  0.81 |    0.01 |       No |
|        addScalar_RawTensor_Torch |    addScalar |      65536 |   float32 |        cpu |      6,550.56 μs |      3,944.540 μs |     216.214 μs |  0.84 |    0.01 |       No |
|           addScalar_Tensor_Torch |    addScalar |      65536 |   float32 |        cpu |      6,790.68 μs |      2,176.844 μs |     119.320 μs |  0.87 |    0.03 |       No |
|    addScalar_RawTensor_Reference |    addScalar |      65536 |   float32 |        cpu |      5,786.08 μs |      2,053.305 μs |     112.549 μs |  0.74 |    0.01 |       No |
|       addScalar_Tensor_Reference |    addScalar |      65536 |   float32 |        cpu |      5,863.34 μs |      1,769.816 μs |      97.010 μs |  0.75 |    0.02 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|             addWithAlpha_PyTorch | addWithAlpha |      65536 |   float32 |        cpu |      6,698.63 μs |        373.091 μs |      20.450 μs |  1.00 |    0.00 |      Yes |
|          addWithAlpha_TorchSharp | addWithAlpha |      65536 |   float32 |        cpu |      5,747.55 μs |      4,134.588 μs |     226.631 μs |  0.86 |    0.03 |       No |
|     addWithAlpha_RawTensor_Torch | addWithAlpha |      65536 |   float32 |        cpu |     11,159.77 μs |      3,785.206 μs |     207.480 μs |  1.67 |    0.03 |       No |
|        addWithAlpha_Tensor_Torch | addWithAlpha |      65536 |   float32 |        cpu |     10,386.33 μs |     21,659.008 μs |   1,187.203 μs |  1.55 |    0.17 |       No |
| addWithAlpha_RawTensor_Reference | addWithAlpha |      65536 |   float32 |        cpu |     13,219.07 μs |      3,593.278 μs |     196.960 μs |  1.97 |    0.03 |       No |
|    addWithAlpha_Tensor_Reference | addWithAlpha |      65536 |   float32 |        cpu |     13,327.08 μs |     17,664.961 μs |     968.276 μs |  1.99 |    0.14 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|               addInPlace_PyTorch |   addInPlace |      65536 |   float32 |        cpu |      6,692.21 μs |        522.589 μs |      28.645 μs |  1.00 |    0.00 |      Yes |
|            addInPlace_TorchSharp |   addInPlace |      65536 |   float32 |        cpu |      2,214.57 μs |      1,024.351 μs |      56.148 μs |  0.33 |    0.01 |       No |
|       addInPlace_RawTensor_Torch |   addInPlace |      65536 |   float32 |        cpu |      6,091.12 μs |      5,736.975 μs |     314.463 μs |  0.91 |    0.05 |       No |
|          addInPlace_Tensor_Torch |   addInPlace |      65536 |   float32 |        cpu |      6,356.23 μs |      2,654.762 μs |     145.516 μs |  0.95 |    0.03 |       No |
|   addInPlace_RawTensor_Reference |   addInPlace |      65536 |   float32 |        cpu |      9,386.46 μs |      6,570.378 μs |     360.145 μs |  1.40 |    0.05 |       No |
|      addInPlace_Tensor_Reference |   addInPlace |      65536 |   float32 |        cpu |     10,191.47 μs |        584.068 μs |      32.015 μs |  1.52 |    0.01 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                   matmul_PyTorch |       matmul |      65536 |   float32 |        cpu |               NA |                NA |             NA |     ? |       ? |      Yes |
|                matmul_TorchSharp |       matmul |      65536 |   float32 |        cpu |      1,592.36 μs |        532.829 μs |      29.206 μs |     ? |       ? |       No |
|           matmul_RawTensor_Torch |       matmul |      65536 |   float32 |        cpu |      1,616.73 μs |      1,219.264 μs |      66.832 μs |     ? |       ? |       No |
|              matmul_Tensor_Torch |       matmul |      65536 |   float32 |        cpu |      1,597.54 μs |        215.876 μs |      11.833 μs |     ? |       ? |       No |
|       matmul_RawTensor_Reference |       matmul |      65536 |   float32 |        cpu |    959,759.97 μs |    193,620.595 μs |  10,612.998 μs |     ? |       ? |       No |
|          matmul_Tensor_Reference |       matmul |      65536 |   float32 |        cpu |    973,320.73 μs |    268,091.456 μs |  14,694.997 μs |     ? |       ? |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|              **fromCpuData_PyTorch** |  **fromCpuData** |      **65536** |   **float32** |       **cuda** |     **32,479.22 μs** |        **297.140 μs** |      **16.287 μs** | **1.000** |    **0.00** |      **Yes** |
|           fromCpuData_TorchSharp |  fromCpuData |      65536 |   float32 |       cuda |         53.77 μs |         51.612 μs |       2.829 μs | 0.002 |    0.00 |       No |
|      fromCpuData_RawTensor_Torch |  fromCpuData |      65536 |   float32 |       cuda |      3,172.77 μs |      3,236.999 μs |     177.431 μs | 0.098 |    0.01 |       No |
|         fromCpuData_Tensor_Torch |  fromCpuData |      65536 |   float32 |       cuda |      3,316.60 μs |     10,257.160 μs |     562.230 μs | 0.102 |    0.02 |       No |
|  fromCpuData_RawTensor_Reference |  fromCpuData |      65536 |   float32 |       cuda |      2,530.05 μs |        239.546 μs |      13.130 μs | 0.078 |    0.00 |       No |
|     fromCpuData_Tensor_Reference |  fromCpuData |      65536 |   float32 |       cuda |      2,543.75 μs |        973.934 μs |      53.385 μs | 0.078 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                    zeros_PyTorch |        zeros |      65536 |   float32 |       cuda |      1,842.16 μs |        270.811 μs |      14.844 μs |  1.00 |    0.00 |      Yes |
|                 zeros_TorchSharp |        zeros |      65536 |   float32 |       cuda |        663.37 μs |        322.023 μs |      17.651 μs |  0.36 |    0.01 |       No |
|            zeros_RawTensor_Torch |        zeros |      65536 |   float32 |       cuda |        756.53 μs |      1,395.940 μs |      76.516 μs |  0.41 |    0.05 |       No |
|               zeros_Tensor_Torch |        zeros |      65536 |   float32 |       cuda |        783.37 μs |        955.068 μs |      52.350 μs |  0.43 |    0.03 |       No |
|        zeros_RawTensor_Reference |        zeros |      65536 |   float32 |       cuda |      3,802.14 μs |      2,811.511 μs |     154.108 μs |  2.06 |    0.07 |       No |
|           zeros_Tensor_Reference |        zeros |      65536 |   float32 |       cuda |      3,886.95 μs |      3,871.452 μs |     212.207 μs |  2.11 |    0.13 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                     ones_PyTorch |         ones |      65536 |   float32 |       cuda |      1,861.89 μs |        210.060 μs |      11.514 μs |  1.00 |    0.00 |      Yes |
|                  ones_TorchSharp |         ones |      65536 |   float32 |       cuda |        644.83 μs |         68.610 μs |       3.761 μs |  0.35 |    0.00 |       No |
|             ones_RawTensor_Torch |         ones |      65536 |   float32 |       cuda |        716.60 μs |        595.618 μs |      32.648 μs |  0.38 |    0.02 |       No |
|                ones_Tensor_Torch |         ones |      65536 |   float32 |       cuda |        828.70 μs |      1,231.805 μs |      67.519 μs |  0.45 |    0.04 |       No |
|         ones_RawTensor_Reference |         ones |      65536 |   float32 |       cuda |      5,874.28 μs |        397.459 μs |      21.786 μs |  3.16 |    0.03 |       No |
|            ones_Tensor_Reference |         ones |      65536 |   float32 |       cuda |      5,689.78 μs |        351.620 μs |      19.273 μs |  3.06 |    0.01 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                     rand_PyTorch |         rand |      65536 |   float32 |       cuda |      1,846.32 μs |        588.729 μs |      32.270 μs |  1.00 |    0.00 |      Yes |
|                  rand_TorchSharp |         rand |      65536 |   float32 |       cuda |        713.43 μs |        298.029 μs |      16.336 μs |  0.39 |    0.00 |       No |
|             rand_RawTensor_Torch |         rand |      65536 |   float32 |       cuda |        949.47 μs |      5,248.182 μs |     287.671 μs |  0.51 |    0.16 |       No |
|                rand_Tensor_Torch |         rand |      65536 |   float32 |       cuda |        861.57 μs |      1,022.601 μs |      56.052 μs |  0.47 |    0.04 |       No |
|         rand_RawTensor_Reference |         rand |      65536 |   float32 |       cuda |     38,059.91 μs |      5,197.159 μs |     284.874 μs | 20.62 |    0.49 |       No |
|            rand_Tensor_Reference |         rand |      65536 |   float32 |       cuda |     37,320.65 μs |      3,705.243 μs |     203.097 μs | 20.22 |    0.28 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                 addition_PyTorch |     addition |      65536 |   float32 |       cuda |      5,823.48 μs |        171.204 μs |       9.384 μs |  1.00 |    0.00 |      Yes |
|              addition_TorchSharp |     addition |      65536 |   float32 |       cuda |        804.23 μs |      1,604.558 μs |      87.951 μs |  0.14 |    0.02 |       No |
|         addition_RawTensor_Torch |     addition |      65536 |   float32 |       cuda |        647.23 μs |        338.677 μs |      18.564 μs |  0.11 |    0.00 |       No |
|            addition_Tensor_Torch |     addition |      65536 |   float32 |       cuda |      1,078.30 μs |      1,574.161 μs |      86.285 μs |  0.19 |    0.01 |       No |
|     addition_RawTensor_Reference |     addition |      65536 |   float32 |       cuda |      9,290.75 μs |        422.416 μs |      23.154 μs |  1.60 |    0.00 |       No |
|        addition_Tensor_Reference |     addition |      65536 |   float32 |       cuda |      9,792.57 μs |      1,619.078 μs |      88.747 μs |  1.68 |    0.01 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                addScalar_PyTorch |    addScalar |      65536 |   float32 |       cuda |      5,781.80 μs |        509.266 μs |      27.915 μs |  1.00 |    0.00 |      Yes |
|             addScalar_TorchSharp |    addScalar |      65536 |   float32 |       cuda |        779.57 μs |        136.917 μs |       7.505 μs |  0.13 |    0.00 |       No |
|        addScalar_RawTensor_Torch |    addScalar |      65536 |   float32 |       cuda |      5,964.93 μs |     14,447.607 μs |     791.922 μs |  1.03 |    0.14 |       No |
|           addScalar_Tensor_Torch |    addScalar |      65536 |   float32 |       cuda |      5,756.17 μs |     15,477.007 μs |     848.347 μs |  1.00 |    0.14 |       No |
|    addScalar_RawTensor_Reference |    addScalar |      65536 |   float32 |       cuda |      5,833.56 μs |        439.017 μs |      24.064 μs |  1.01 |    0.01 |       No |
|       addScalar_Tensor_Reference |    addScalar |      65536 |   float32 |       cuda |      5,889.14 μs |      1,280.059 μs |      70.164 μs |  1.02 |    0.01 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|             addWithAlpha_PyTorch | addWithAlpha |      65536 |   float32 |       cuda |      4,997.13 μs |         69.001 μs |       3.782 μs |  1.00 |    0.00 |      Yes |
|          addWithAlpha_TorchSharp | addWithAlpha |      65536 |   float32 |       cuda |        590.07 μs |        333.747 μs |      18.294 μs |  0.12 |    0.00 |       No |
|     addWithAlpha_RawTensor_Torch | addWithAlpha |      65536 |   float32 |       cuda |      1,648.50 μs |      2,777.948 μs |     152.269 μs |  0.33 |    0.03 |       No |
|        addWithAlpha_Tensor_Torch | addWithAlpha |      65536 |   float32 |       cuda |      2,164.00 μs |      4,194.504 μs |     229.915 μs |  0.43 |    0.05 |       No |
| addWithAlpha_RawTensor_Reference | addWithAlpha |      65536 |   float32 |       cuda |     12,544.28 μs |      1,294.846 μs |      70.975 μs |  2.51 |    0.01 |       No |
|    addWithAlpha_Tensor_Reference | addWithAlpha |      65536 |   float32 |       cuda |     13,593.18 μs |      2,598.550 μs |     142.435 μs |  2.72 |    0.03 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|               addInPlace_PyTorch |   addInPlace |      65536 |   float32 |       cuda |      4,996.11 μs |         52.427 μs |       2.874 μs |  1.00 |    0.00 |      Yes |
|            addInPlace_TorchSharp |   addInPlace |      65536 |   float32 |       cuda |        443.57 μs |        736.094 μs |      40.348 μs |  0.09 |    0.01 |       No |
|       addInPlace_RawTensor_Torch |   addInPlace |      65536 |   float32 |       cuda |        724.77 μs |      1,038.826 μs |      56.942 μs |  0.15 |    0.01 |       No |
|          addInPlace_Tensor_Torch |   addInPlace |      65536 |   float32 |       cuda |        936.80 μs |        985.166 μs |      54.000 μs |  0.19 |    0.01 |       No |
|   addInPlace_RawTensor_Reference |   addInPlace |      65536 |   float32 |       cuda |     10,253.67 μs |      2,927.213 μs |     160.450 μs |  2.05 |    0.03 |       No |
|      addInPlace_Tensor_Reference |   addInPlace |      65536 |   float32 |       cuda |      9,613.65 μs |        576.901 μs |      31.622 μs |  1.92 |    0.01 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                   matmul_PyTorch |       matmul |      65536 |   float32 |       cuda |               NA |                NA |             NA |     ? |       ? |      Yes |
|                matmul_TorchSharp |       matmul |      65536 |   float32 |       cuda |        404.57 μs |        524.929 μs |      28.773 μs |     ? |       ? |       No |
|           matmul_RawTensor_Torch |       matmul |      65536 |   float32 |       cuda |        353.07 μs |      1,196.967 μs |      65.610 μs |     ? |       ? |       No |
|              matmul_Tensor_Torch |       matmul |      65536 |   float32 |       cuda |        394.97 μs |        863.532 μs |      47.333 μs |     ? |       ? |       No |
|       matmul_RawTensor_Reference |       matmul |      65536 |   float32 |       cuda |  1,003,106.83 μs |     64,674.595 μs |   3,545.033 μs |     ? |       ? |       No |
|          matmul_Tensor_Reference |       matmul |      65536 |   float32 |       cuda |  1,027,958.10 μs |    463,117.092 μs |  25,385.011 μs |     ? |       ? |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|              **fromCpuData_PyTorch** |  **fromCpuData** |      **65536** |   **float64** |        **cpu** |     **30,999.83 μs** |         **19.877 μs** |       **1.090 μs** | **1.000** |    **0.00** |      **Yes** |
|           fromCpuData_TorchSharp |  fromCpuData |      65536 |   float64 |        cpu |         44.38 μs |         20.328 μs |       1.114 μs | 0.001 |    0.00 |       No |
|      fromCpuData_RawTensor_Torch |  fromCpuData |      65536 |   float64 |        cpu |      2,497.86 μs |        639.352 μs |      35.045 μs | 0.081 |    0.00 |       No |
|         fromCpuData_Tensor_Torch |  fromCpuData |      65536 |   float64 |        cpu |      2,355.67 μs |        797.861 μs |      43.733 μs | 0.076 |    0.00 |       No |
|  fromCpuData_RawTensor_Reference |  fromCpuData |      65536 |   float64 |        cpu |      2,687.20 μs |        245.816 μs |      13.474 μs | 0.087 |    0.00 |       No |
|     fromCpuData_Tensor_Reference |  fromCpuData |      65536 |   float64 |        cpu |      2,725.55 μs |      1,236.925 μs |      67.800 μs | 0.088 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                    zeros_PyTorch |        zeros |      65536 |   float64 |        cpu |      2,721.68 μs |        323.148 μs |      17.713 μs |  1.00 |    0.00 |      Yes |
|                 zeros_TorchSharp |        zeros |      65536 |   float64 |        cpu |      4,834.03 μs |      1,107.663 μs |      60.715 μs |  1.78 |    0.02 |       No |
|            zeros_RawTensor_Torch |        zeros |      65536 |   float64 |        cpu |      8,771.23 μs |        681.402 μs |      37.350 μs |  3.22 |    0.02 |       No |
|               zeros_Tensor_Torch |        zeros |      65536 |   float64 |        cpu |      8,556.73 μs |      6,317.592 μs |     346.289 μs |  3.14 |    0.14 |       No |
|        zeros_RawTensor_Reference |        zeros |      65536 |   float64 |        cpu |      4,543.24 μs |      1,146.722 μs |      62.856 μs |  1.67 |    0.02 |       No |
|           zeros_Tensor_Reference |        zeros |      65536 |   float64 |        cpu |      4,656.89 μs |      2,060.823 μs |     112.961 μs |  1.71 |    0.05 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                     ones_PyTorch |         ones |      65536 |   float64 |        cpu |      2,787.55 μs |        467.312 μs |      25.615 μs |  1.00 |    0.00 |      Yes |
|                  ones_TorchSharp |         ones |      65536 |   float64 |        cpu |      4,772.05 μs |      1,833.320 μs |     100.490 μs |  1.71 |    0.03 |       No |
|             ones_RawTensor_Torch |         ones |      65536 |   float64 |        cpu |      8,453.13 μs |      1,465.467 μs |      80.327 μs |  3.03 |    0.02 |       No |
|                ones_Tensor_Torch |         ones |      65536 |   float64 |        cpu |      8,619.40 μs |      1,257.386 μs |      68.922 μs |  3.09 |    0.01 |       No |
|         ones_RawTensor_Reference |         ones |      65536 |   float64 |        cpu |      7,281.16 μs |        569.552 μs |      31.219 μs |  2.61 |    0.02 |       No |
|            ones_Tensor_Reference |         ones |      65536 |   float64 |        cpu |      7,426.03 μs |      2,495.169 μs |     136.769 μs |  2.66 |    0.07 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                     rand_PyTorch |         rand |      65536 |   float64 |        cpu |     39,516.17 μs |         92.395 μs |       5.064 μs |  1.00 |    0.00 |      Yes |
|                  rand_TorchSharp |         rand |      65536 |   float64 |        cpu |     21,412.04 μs |      2,755.318 μs |     151.028 μs |  0.54 |    0.00 |       No |
|             rand_RawTensor_Torch |         rand |      65536 |   float64 |        cpu |     44,886.98 μs |        269.670 μs |      14.782 μs |  1.14 |    0.00 |       No |
|                rand_Tensor_Torch |         rand |      65536 |   float64 |        cpu |     44,723.76 μs |     13,014.552 μs |     713.371 μs |  1.13 |    0.02 |       No |
|         rand_RawTensor_Reference |         rand |      65536 |   float64 |        cpu |     37,721.10 μs |     15,232.178 μs |     834.927 μs |  0.95 |    0.02 |       No |
|            rand_Tensor_Reference |         rand |      65536 |   float64 |        cpu |     40,460.80 μs |      5,924.818 μs |     324.759 μs |  1.02 |    0.01 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                 addition_PyTorch |     addition |      65536 |   float64 |        cpu |     10,596.82 μs |        553.945 μs |      30.364 μs |  1.00 |    0.00 |      Yes |
|              addition_TorchSharp |     addition |      65536 |   float64 |        cpu |      5,874.93 μs |      4,398.692 μs |     241.107 μs |  0.55 |    0.02 |       No |
|         addition_RawTensor_Torch |     addition |      65536 |   float64 |        cpu |      5,915.60 μs |      1,785.614 μs |      97.876 μs |  0.56 |    0.01 |       No |
|            addition_Tensor_Torch |     addition |      65536 |   float64 |        cpu |      6,226.74 μs |      1,510.202 μs |      82.779 μs |  0.59 |    0.01 |       No |
|     addition_RawTensor_Reference |     addition |      65536 |   float64 |        cpu |     10,785.66 μs |     11,951.321 μs |     655.092 μs |  1.02 |    0.06 |       No |
|        addition_Tensor_Reference |     addition |      65536 |   float64 |        cpu |      9,746.52 μs |      7,171.145 μs |     393.075 μs |  0.92 |    0.04 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                addScalar_PyTorch |    addScalar |      65536 |   float64 |        cpu |     10,581.08 μs |      2,208.150 μs |     121.036 μs |  1.00 |    0.00 |      Yes |
|             addScalar_TorchSharp |    addScalar |      65536 |   float64 |        cpu |      6,454.88 μs |      1,920.947 μs |     105.294 μs |  0.61 |    0.02 |       No |
|        addScalar_RawTensor_Torch |    addScalar |      65536 |   float64 |        cpu |      6,463.08 μs |      1,669.954 μs |      91.536 μs |  0.61 |    0.01 |       No |
|           addScalar_Tensor_Torch |    addScalar |      65536 |   float64 |        cpu |      6,999.06 μs |      3,352.478 μs |     183.761 μs |  0.66 |    0.02 |       No |
|    addScalar_RawTensor_Reference |    addScalar |      65536 |   float64 |        cpu |      5,791.90 μs |        709.370 μs |      38.883 μs |  0.55 |    0.01 |       No |
|       addScalar_Tensor_Reference |    addScalar |      65536 |   float64 |        cpu |      5,966.95 μs |      2,377.169 μs |     130.301 μs |  0.56 |    0.01 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|             addWithAlpha_PyTorch | addWithAlpha |      65536 |   float64 |        cpu |      9,587.64 μs |        886.327 μs |      48.583 μs |  1.00 |    0.00 |      Yes |
|          addWithAlpha_TorchSharp | addWithAlpha |      65536 |   float64 |        cpu |      5,871.75 μs |      4,401.470 μs |     241.259 μs |  0.61 |    0.03 |       No |
|     addWithAlpha_RawTensor_Torch | addWithAlpha |      65536 |   float64 |        cpu |     11,469.61 μs |      7,831.049 μs |     429.246 μs |  1.20 |    0.05 |       No |
|        addWithAlpha_Tensor_Torch | addWithAlpha |      65536 |   float64 |        cpu |     13,251.63 μs |      4,592.556 μs |     251.733 μs |  1.38 |    0.03 |       No |
| addWithAlpha_RawTensor_Reference | addWithAlpha |      65536 |   float64 |        cpu |     12,560.79 μs |      8,781.350 μs |     481.335 μs |  1.31 |    0.05 |       No |
|    addWithAlpha_Tensor_Reference | addWithAlpha |      65536 |   float64 |        cpu |     13,049.76 μs |      7,751.238 μs |     424.872 μs |  1.36 |    0.05 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|               addInPlace_PyTorch |   addInPlace |      65536 |   float64 |        cpu |      8,633.57 μs |        699.660 μs |      38.351 μs |  1.00 |    0.00 |      Yes |
|            addInPlace_TorchSharp |   addInPlace |      65536 |   float64 |        cpu |      2,365.81 μs |      4,415.746 μs |     242.042 μs |  0.27 |    0.03 |       No |
|       addInPlace_RawTensor_Torch |   addInPlace |      65536 |   float64 |        cpu |      5,881.85 μs |      3,803.142 μs |     208.463 μs |  0.68 |    0.02 |       No |
|          addInPlace_Tensor_Torch |   addInPlace |      65536 |   float64 |        cpu |      6,197.55 μs |      4,870.913 μs |     266.991 μs |  0.72 |    0.03 |       No |
|   addInPlace_RawTensor_Reference |   addInPlace |      65536 |   float64 |        cpu |     10,084.61 μs |        778.036 μs |      42.647 μs |  1.17 |    0.00 |       No |
|      addInPlace_Tensor_Reference |   addInPlace |      65536 |   float64 |        cpu |     10,337.22 μs |      2,368.431 μs |     129.822 μs |  1.20 |    0.02 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                   matmul_PyTorch |       matmul |      65536 |   float64 |        cpu |               NA |                NA |             NA |     ? |       ? |      Yes |
|                matmul_TorchSharp |       matmul |      65536 |   float64 |        cpu |      3,135.86 μs |      1,524.098 μs |      83.541 μs |     ? |       ? |       No |
|           matmul_RawTensor_Torch |       matmul |      65536 |   float64 |        cpu |      3,292.88 μs |      2,019.337 μs |     110.687 μs |     ? |       ? |       No |
|              matmul_Tensor_Torch |       matmul |      65536 |   float64 |        cpu |      3,363.77 μs |      1,195.937 μs |      65.553 μs |     ? |       ? |       No |
|       matmul_RawTensor_Reference |       matmul |      65536 |   float64 |        cpu |  1,021,613.33 μs |    103,993.996 μs |   5,700.262 μs |     ? |       ? |       No |
|          matmul_Tensor_Reference |       matmul |      65536 |   float64 |        cpu |  1,016,118.87 μs |     35,677.383 μs |   1,955.598 μs |     ? |       ? |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|              **fromCpuData_PyTorch** |  **fromCpuData** |      **65536** |   **float64** |       **cuda** |     **32,608.71 μs** |      **1,119.920 μs** |      **61.387 μs** | **1.000** |    **0.00** |      **Yes** |
|           fromCpuData_TorchSharp |  fromCpuData |      65536 |   float64 |       cuda |         54.53 μs |         50.504 μs |       2.768 μs | 0.002 |    0.00 |       No |
|      fromCpuData_RawTensor_Torch |  fromCpuData |      65536 |   float64 |       cuda |      5,325.33 μs |     32,873.148 μs |   1,801.888 μs | 0.163 |    0.06 |       No |
|         fromCpuData_Tensor_Torch |  fromCpuData |      65536 |   float64 |       cuda |      4,289.27 μs |      1,584.299 μs |      86.841 μs | 0.132 |    0.00 |       No |
|  fromCpuData_RawTensor_Reference |  fromCpuData |      65536 |   float64 |       cuda |      2,674.19 μs |      1,913.377 μs |     104.879 μs | 0.082 |    0.00 |       No |
|     fromCpuData_Tensor_Reference |  fromCpuData |      65536 |   float64 |       cuda |      2,672.34 μs |        177.873 μs |       9.750 μs | 0.082 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                    zeros_PyTorch |        zeros |      65536 |   float64 |       cuda |      1,821.58 μs |        182.840 μs |      10.022 μs |  1.00 |    0.00 |      Yes |
|                 zeros_TorchSharp |        zeros |      65536 |   float64 |       cuda |        689.23 μs |      1,057.466 μs |      57.963 μs |  0.38 |    0.03 |       No |
|            zeros_RawTensor_Torch |        zeros |      65536 |   float64 |       cuda |        741.07 μs |        142.047 μs |       7.786 μs |  0.41 |    0.01 |       No |
|               zeros_Tensor_Torch |        zeros |      65536 |   float64 |       cuda |        812.03 μs |      1,164.722 μs |      63.842 μs |  0.45 |    0.03 |       No |
|        zeros_RawTensor_Reference |        zeros |      65536 |   float64 |       cuda |      4,604.05 μs |      3,102.480 μs |     170.057 μs |  2.53 |    0.08 |       No |
|           zeros_Tensor_Reference |        zeros |      65536 |   float64 |       cuda |      4,723.67 μs |      6,307.532 μs |     345.737 μs |  2.59 |    0.20 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                     ones_PyTorch |         ones |      65536 |   float64 |       cuda |      1,841.66 μs |        272.035 μs |      14.911 μs |  1.00 |    0.00 |      Yes |
|                  ones_TorchSharp |         ones |      65536 |   float64 |       cuda |        671.20 μs |        287.817 μs |      15.776 μs |  0.36 |    0.01 |       No |
|             ones_RawTensor_Torch |         ones |      65536 |   float64 |       cuda |        740.37 μs |        963.905 μs |      52.835 μs |  0.40 |    0.03 |       No |
|                ones_Tensor_Torch |         ones |      65536 |   float64 |       cuda |        700.03 μs |        173.808 μs |       9.527 μs |  0.38 |    0.01 |       No |
|         ones_RawTensor_Reference |         ones |      65536 |   float64 |       cuda |      7,305.97 μs |        477.568 μs |      26.177 μs |  3.97 |    0.04 |       No |
|            ones_Tensor_Reference |         ones |      65536 |   float64 |       cuda |      7,274.29 μs |      1,377.665 μs |      75.514 μs |  3.95 |    0.05 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                     rand_PyTorch |         rand |      65536 |   float64 |       cuda |      1,853.79 μs |        200.391 μs |      10.984 μs |  1.00 |    0.00 |      Yes |
|                  rand_TorchSharp |         rand |      65536 |   float64 |       cuda |        697.57 μs |        516.140 μs |      28.291 μs |  0.38 |    0.02 |       No |
|             rand_RawTensor_Torch |         rand |      65536 |   float64 |       cuda |        890.50 μs |      5,087.512 μs |     278.864 μs |  0.48 |    0.15 |       No |
|                rand_Tensor_Torch |         rand |      65536 |   float64 |       cuda |        759.50 μs |        587.473 μs |      32.201 μs |  0.41 |    0.02 |       No |
|         rand_RawTensor_Reference |         rand |      65536 |   float64 |       cuda |     37,397.97 μs |      4,066.440 μs |     222.895 μs | 20.17 |    0.23 |       No |
|            rand_Tensor_Reference |         rand |      65536 |   float64 |       cuda |     37,707.32 μs |      2,558.114 μs |     140.219 μs | 20.34 |    0.05 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                 addition_PyTorch |     addition |      65536 |   float64 |       cuda |      5,680.28 μs |        457.290 μs |      25.066 μs |  1.00 |    0.00 |      Yes |
|              addition_TorchSharp |     addition |      65536 |   float64 |       cuda |        617.90 μs |        694.403 μs |      38.063 μs |  0.11 |    0.01 |       No |
|         addition_RawTensor_Torch |     addition |      65536 |   float64 |       cuda |        622.47 μs |        195.407 μs |      10.711 μs |  0.11 |    0.00 |       No |
|            addition_Tensor_Torch |     addition |      65536 |   float64 |       cuda |        924.93 μs |        494.645 μs |      27.113 μs |  0.16 |    0.01 |       No |
|     addition_RawTensor_Reference |     addition |      65536 |   float64 |       cuda |      9,527.21 μs |      3,588.014 μs |     196.671 μs |  1.68 |    0.04 |       No |
|        addition_Tensor_Reference |     addition |      65536 |   float64 |       cuda |     10,883.44 μs |      9,306.653 μs |     510.129 μs |  1.92 |    0.10 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                addScalar_PyTorch |    addScalar |      65536 |   float64 |       cuda |      5,686.37 μs |      1,223.449 μs |      67.061 μs |  1.00 |    0.00 |      Yes |
|             addScalar_TorchSharp |    addScalar |      65536 |   float64 |       cuda |        725.73 μs |        391.358 μs |      21.452 μs |  0.13 |    0.01 |       No |
|        addScalar_RawTensor_Torch |    addScalar |      65536 |   float64 |       cuda |      4,569.97 μs |      7,235.947 μs |     396.627 μs |  0.80 |    0.08 |       No |
|           addScalar_Tensor_Torch |    addScalar |      65536 |   float64 |       cuda |      5,029.77 μs |     11,509.169 μs |     630.856 μs |  0.88 |    0.10 |       No |
|    addScalar_RawTensor_Reference |    addScalar |      65536 |   float64 |       cuda |      5,819.70 μs |      1,009.035 μs |      55.309 μs |  1.02 |    0.02 |       No |
|       addScalar_Tensor_Reference |    addScalar |      65536 |   float64 |       cuda |      5,960.74 μs |      2,159.135 μs |     118.349 μs |  1.05 |    0.03 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|             addWithAlpha_PyTorch | addWithAlpha |      65536 |   float64 |       cuda |      4,728.05 μs |      1,401.765 μs |      76.835 μs |  1.00 |    0.00 |      Yes |
|          addWithAlpha_TorchSharp | addWithAlpha |      65536 |   float64 |       cuda |        581.23 μs |         24.903 μs |       1.365 μs |  0.12 |    0.00 |       No |
|     addWithAlpha_RawTensor_Torch | addWithAlpha |      65536 |   float64 |       cuda |      1,475.43 μs |        557.794 μs |      30.575 μs |  0.31 |    0.01 |       No |
|        addWithAlpha_Tensor_Torch | addWithAlpha |      65536 |   float64 |       cuda |      1,979.30 μs |        193.452 μs |      10.604 μs |  0.42 |    0.01 |       No |
| addWithAlpha_RawTensor_Reference | addWithAlpha |      65536 |   float64 |       cuda |     13,586.87 μs |      7,987.913 μs |     437.844 μs |  2.87 |    0.06 |       No |
|    addWithAlpha_Tensor_Reference | addWithAlpha |      65536 |   float64 |       cuda |     12,828.24 μs |      1,795.736 μs |      98.430 μs |  2.71 |    0.04 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|               addInPlace_PyTorch |   addInPlace |      65536 |   float64 |       cuda |      4,738.53 μs |      1,109.138 μs |      60.796 μs |  1.00 |    0.00 |      Yes |
|            addInPlace_TorchSharp |   addInPlace |      65536 |   float64 |       cuda |        359.87 μs |        468.265 μs |      25.667 μs |  0.08 |    0.00 |       No |
|       addInPlace_RawTensor_Torch |   addInPlace |      65536 |   float64 |       cuda |        710.67 μs |      2,265.799 μs |     124.196 μs |  0.15 |    0.03 |       No |
|          addInPlace_Tensor_Torch |   addInPlace |      65536 |   float64 |       cuda |        862.97 μs |         90.749 μs |       4.974 μs |  0.18 |    0.00 |       No |
|   addInPlace_RawTensor_Reference |   addInPlace |      65536 |   float64 |       cuda |     10,197.55 μs |      2,645.501 μs |     145.009 μs |  2.15 |    0.03 |       No |
|      addInPlace_Tensor_Reference |   addInPlace |      65536 |   float64 |       cuda |     10,379.14 μs |      2,630.014 μs |     144.160 μs |  2.19 |    0.05 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                   matmul_PyTorch |       matmul |      65536 |   float64 |       cuda |               NA |                NA |             NA |     ? |       ? |      Yes |
|                matmul_TorchSharp |       matmul |      65536 |   float64 |       cuda |        149.53 μs |        708.557 μs |      38.838 μs |     ? |       ? |       No |
|           matmul_RawTensor_Torch |       matmul |      65536 |   float64 |       cuda |        111.60 μs |        478.514 μs |      26.229 μs |     ? |       ? |       No |
|              matmul_Tensor_Torch |       matmul |      65536 |   float64 |       cuda |        141.67 μs |        488.068 μs |      26.753 μs |     ? |       ? |       No |
|       matmul_RawTensor_Reference |       matmul |      65536 |   float64 |       cuda |  1,015,215.23 μs |     19,602.445 μs |   1,074.476 μs |     ? |       ? |       No |
|          matmul_Tensor_Reference |       matmul |      65536 |   float64 |       cuda |  1,012,842.90 μs |    107,480.929 μs |   5,891.392 μs |     ? |       ? |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|              **fromCpuData_PyTorch** |  **fromCpuData** |      **65536** |     **int32** |        **cpu** |     **24,502.71 μs** |        **769.830 μs** |      **42.197 μs** | **1.000** |    **0.00** |      **Yes** |
|           fromCpuData_TorchSharp |  fromCpuData |      65536 |     int32 |        cpu |         42.59 μs |         18.546 μs |       1.017 μs | 0.002 |    0.00 |       No |
|      fromCpuData_RawTensor_Torch |  fromCpuData |      65536 |     int32 |        cpu |      1,716.64 μs |        575.905 μs |      31.567 μs | 0.070 |    0.00 |       No |
|         fromCpuData_Tensor_Torch |  fromCpuData |      65536 |     int32 |        cpu |      1,699.67 μs |        106.186 μs |       5.820 μs | 0.069 |    0.00 |       No |
|  fromCpuData_RawTensor_Reference |  fromCpuData |      65536 |     int32 |        cpu |      2,407.25 μs |        625.455 μs |      34.283 μs | 0.098 |    0.00 |       No |
|     fromCpuData_Tensor_Reference |  fromCpuData |      65536 |     int32 |        cpu |      2,319.44 μs |        465.287 μs |      25.504 μs | 0.095 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                    zeros_PyTorch |        zeros |      65536 |     int32 |        cpu |      1,860.07 μs |        274.094 μs |      15.024 μs |  1.00 |    0.00 |      Yes |
|                 zeros_TorchSharp |        zeros |      65536 |     int32 |        cpu |      4,687.60 μs |        921.436 μs |      50.507 μs |  2.52 |    0.04 |       No |
|            zeros_RawTensor_Torch |        zeros |      65536 |     int32 |        cpu |      4,787.33 μs |      2,378.259 μs |     130.360 μs |  2.57 |    0.05 |       No |
|               zeros_Tensor_Torch |        zeros |      65536 |     int32 |        cpu |      4,707.49 μs |      1,542.982 μs |      84.576 μs |  2.53 |    0.06 |       No |
|        zeros_RawTensor_Reference |        zeros |      65536 |     int32 |        cpu |      3,554.76 μs |        720.605 μs |      39.499 μs |  1.91 |    0.01 |       No |
|           zeros_Tensor_Reference |        zeros |      65536 |     int32 |        cpu |      3,539.54 μs |        553.904 μs |      30.361 μs |  1.90 |    0.02 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                     ones_PyTorch |         ones |      65536 |     int32 |        cpu |      1,840.50 μs |        231.771 μs |      12.704 μs |  1.00 |    0.00 |      Yes |
|                  ones_TorchSharp |         ones |      65536 |     int32 |        cpu |      4,702.05 μs |      2,436.456 μs |     133.550 μs |  2.56 |    0.09 |       No |
|             ones_RawTensor_Torch |         ones |      65536 |     int32 |        cpu |      4,769.62 μs |      2,182.051 μs |     119.606 μs |  2.59 |    0.08 |       No |
|                ones_Tensor_Torch |         ones |      65536 |     int32 |        cpu |      4,645.54 μs |      3,614.244 μs |     198.109 μs |  2.52 |    0.13 |       No |
|         ones_RawTensor_Reference |         ones |      65536 |     int32 |        cpu |      5,654.45 μs |      2,639.662 μs |     144.689 μs |  3.07 |    0.10 |       No |
|            ones_Tensor_Reference |         ones |      65536 |     int32 |        cpu |      5,533.37 μs |      3,212.481 μs |     176.087 μs |  3.01 |    0.09 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                     rand_PyTorch |         rand |      65536 |     int32 |        cpu |               NA |                NA |             NA |     ? |       ? |      Yes |
|                  rand_TorchSharp |         rand |      65536 |     int32 |        cpu |     34,342.94 μs |      4,807.620 μs |     263.522 μs |     ? |       ? |       No |
|             rand_RawTensor_Torch |         rand |      65536 |     int32 |        cpu |               NA |                NA |             NA |     ? |       ? |       No |
|                rand_Tensor_Torch |         rand |      65536 |     int32 |        cpu |               NA |                NA |             NA |     ? |       ? |       No |
|         rand_RawTensor_Reference |         rand |      65536 |     int32 |        cpu |               NA |                NA |             NA |     ? |       ? |       No |
|            rand_Tensor_Reference |         rand |      65536 |     int32 |        cpu |               NA |                NA |             NA |     ? |       ? |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                 addition_PyTorch |     addition |      65536 |     int32 |        cpu |      6,664.71 μs |        439.644 μs |      24.098 μs |  1.00 |    0.00 |      Yes |
|              addition_TorchSharp |     addition |      65536 |     int32 |        cpu |      5,803.67 μs |      3,711.698 μs |     203.451 μs |  0.87 |    0.03 |       No |
|         addition_RawTensor_Torch |     addition |      65536 |     int32 |        cpu |      5,893.41 μs |      4,144.281 μs |     227.162 μs |  0.88 |    0.03 |       No |
|            addition_Tensor_Torch |     addition |      65536 |     int32 |        cpu |      6,115.30 μs |      2,479.598 μs |     135.915 μs |  0.92 |    0.02 |       No |
|     addition_RawTensor_Reference |     addition |      65536 |     int32 |        cpu |      9,830.41 μs |        738.743 μs |      40.493 μs |  1.47 |    0.00 |       No |
|        addition_Tensor_Reference |     addition |      65536 |     int32 |        cpu |     10,396.68 μs |      9,213.467 μs |     505.021 μs |  1.56 |    0.08 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                addScalar_PyTorch |    addScalar |      65536 |     int32 |        cpu |      7,680.86 μs |        389.951 μs |      21.375 μs |  1.00 |    0.00 |      Yes |
|             addScalar_TorchSharp |    addScalar |      65536 |     int32 |        cpu |      6,202.75 μs |      1,279.245 μs |      70.120 μs |  0.81 |    0.01 |       No |
|        addScalar_RawTensor_Torch |    addScalar |      65536 |     int32 |        cpu |      8,132.95 μs |      2,680.491 μs |     146.927 μs |  1.06 |    0.02 |       No |
|           addScalar_Tensor_Torch |    addScalar |      65536 |     int32 |        cpu |     11,422.94 μs |      6,839.608 μs |     374.902 μs |  1.49 |    0.05 |       No |
|    addScalar_RawTensor_Reference |    addScalar |      65536 |     int32 |        cpu |               NA |                NA |             NA |     ? |       ? |       No |
|       addScalar_Tensor_Reference |    addScalar |      65536 |     int32 |        cpu |    129,400.10 μs |    169,875.471 μs |   9,311.448 μs | 16.85 |    1.25 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|             addWithAlpha_PyTorch | addWithAlpha |      65536 |     int32 |        cpu |      5,728.29 μs |        365.742 μs |      20.048 μs |  1.00 |    0.00 |      Yes |
|          addWithAlpha_TorchSharp | addWithAlpha |      65536 |     int32 |        cpu |      5,784.55 μs |        771.537 μs |      42.291 μs |  1.01 |    0.01 |       No |
|     addWithAlpha_RawTensor_Torch | addWithAlpha |      65536 |     int32 |        cpu |     15,304.95 μs |      7,612.913 μs |     417.289 μs |  2.67 |    0.07 |       No |
|        addWithAlpha_Tensor_Torch | addWithAlpha |      65536 |     int32 |        cpu |     24,002.47 μs |     15,776.580 μs |     864.768 μs |  4.19 |    0.15 |       No |
| addWithAlpha_RawTensor_Reference | addWithAlpha |      65536 |     int32 |        cpu |               NA |                NA |             NA |     ? |       ? |       No |
|    addWithAlpha_Tensor_Reference | addWithAlpha |      65536 |     int32 |        cpu |    240,134.23 μs |     47,350.209 μs |   2,595.425 μs | 41.92 |    0.55 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|               addInPlace_PyTorch |   addInPlace |      65536 |     int32 |        cpu |      5,704.01 μs |        427.261 μs |      23.420 μs |  1.00 |    0.00 |      Yes |
|            addInPlace_TorchSharp |   addInPlace |      65536 |     int32 |        cpu |      2,024.41 μs |      2,628.721 μs |     144.089 μs |  0.35 |    0.03 |       No |
|       addInPlace_RawTensor_Torch |   addInPlace |      65536 |     int32 |        cpu |      5,896.94 μs |      3,277.375 μs |     179.644 μs |  1.03 |    0.04 |       No |
|          addInPlace_Tensor_Torch |   addInPlace |      65536 |     int32 |        cpu |      7,104.43 μs |      6,312.681 μs |     346.019 μs |  1.25 |    0.06 |       No |
|   addInPlace_RawTensor_Reference |   addInPlace |      65536 |     int32 |        cpu |     10,529.75 μs |     21,562.165 μs |   1,181.895 μs |  1.85 |    0.20 |       No |
|      addInPlace_Tensor_Reference |   addInPlace |      65536 |     int32 |        cpu |      9,382.94 μs |      1,489.606 μs |      81.650 μs |  1.65 |    0.02 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                   matmul_PyTorch |       matmul |      65536 |     int32 |        cpu |               NA |                NA |             NA |     ? |       ? |      Yes |
|                matmul_TorchSharp |       matmul |      65536 |     int32 |        cpu |     34,447.16 μs |      5,412.427 μs |     296.673 μs |     ? |       ? |       No |
|           matmul_RawTensor_Torch |       matmul |      65536 |     int32 |        cpu |     34,958.09 μs |     14,539.355 μs |     796.951 μs |     ? |       ? |       No |
|              matmul_Tensor_Torch |       matmul |      65536 |     int32 |        cpu |     35,929.55 μs |     11,047.511 μs |     605.551 μs |     ? |       ? |       No |
|       matmul_RawTensor_Reference |       matmul |      65536 |     int32 |        cpu |  1,009,555.80 μs |     26,938.568 μs |   1,476.594 μs |     ? |       ? |       No |
|          matmul_Tensor_Reference |       matmul |      65536 |     int32 |        cpu |  1,001,072.53 μs |    341,119.168 μs |  18,697.893 μs |     ? |       ? |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|              **fromCpuData_PyTorch** |  **fromCpuData** |      **65536** |     **int32** |       **cuda** |     **25,540.34 μs** |      **1,199.453 μs** |      **65.746 μs** | **1.000** |    **0.00** |      **Yes** |
|           fromCpuData_TorchSharp |  fromCpuData |      65536 |     int32 |       cuda |         56.50 μs |        165.325 μs |       9.062 μs | 0.002 |    0.00 |       No |
|      fromCpuData_RawTensor_Torch |  fromCpuData |      65536 |     int32 |       cuda |      2,854.03 μs |      1,971.989 μs |     108.091 μs | 0.112 |    0.00 |       No |
|         fromCpuData_Tensor_Torch |  fromCpuData |      65536 |     int32 |       cuda |      2,881.50 μs |      1,965.979 μs |     107.762 μs | 0.113 |    0.00 |       No |
|  fromCpuData_RawTensor_Reference |  fromCpuData |      65536 |     int32 |       cuda |      2,406.20 μs |        139.507 μs |       7.647 μs | 0.094 |    0.00 |       No |
|     fromCpuData_Tensor_Reference |  fromCpuData |      65536 |     int32 |       cuda |      2,377.82 μs |      1,186.983 μs |      65.063 μs | 0.093 |    0.00 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                    zeros_PyTorch |        zeros |      65536 |     int32 |       cuda |      1,861.87 μs |        441.891 μs |      24.222 μs |  1.00 |    0.00 |      Yes |
|                 zeros_TorchSharp |        zeros |      65536 |     int32 |       cuda |        653.07 μs |        267.595 μs |      14.668 μs |  0.35 |    0.01 |       No |
|            zeros_RawTensor_Torch |        zeros |      65536 |     int32 |       cuda |        772.80 μs |        995.292 μs |      54.555 μs |  0.42 |    0.03 |       No |
|               zeros_Tensor_Torch |        zeros |      65536 |     int32 |       cuda |        728.27 μs |      1,027.033 μs |      56.295 μs |  0.39 |    0.03 |       No |
|        zeros_RawTensor_Reference |        zeros |      65536 |     int32 |       cuda |      3,707.78 μs |      3,263.743 μs |     178.897 μs |  1.99 |    0.09 |       No |
|           zeros_Tensor_Reference |        zeros |      65536 |     int32 |       cuda |      3,655.83 μs |      1,085.490 μs |      59.499 μs |  1.96 |    0.05 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                     ones_PyTorch |         ones |      65536 |     int32 |       cuda |      1,856.09 μs |        277.290 μs |      15.199 μs |  1.00 |    0.00 |      Yes |
|                  ones_TorchSharp |         ones |      65536 |     int32 |       cuda |        642.47 μs |        200.426 μs |      10.986 μs |  0.35 |    0.01 |       No |
|             ones_RawTensor_Torch |         ones |      65536 |     int32 |       cuda |        664.93 μs |        147.541 μs |       8.087 μs |  0.36 |    0.01 |       No |
|                ones_Tensor_Torch |         ones |      65536 |     int32 |       cuda |        687.10 μs |         13.774 μs |       0.755 μs |  0.37 |    0.00 |       No |
|         ones_RawTensor_Reference |         ones |      65536 |     int32 |       cuda |      5,610.58 μs |      2,422.418 μs |     132.781 μs |  3.02 |    0.09 |       No |
|            ones_Tensor_Reference |         ones |      65536 |     int32 |       cuda |      5,678.94 μs |      4,517.008 μs |     247.592 μs |  3.06 |    0.11 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                     rand_PyTorch |         rand |      65536 |     int32 |       cuda |               NA |                NA |             NA |     ? |       ? |      Yes |
|                  rand_TorchSharp |         rand |      65536 |     int32 |       cuda |        785.77 μs |        685.921 μs |      37.598 μs |     ? |       ? |       No |
|             rand_RawTensor_Torch |         rand |      65536 |     int32 |       cuda |               NA |                NA |             NA |     ? |       ? |       No |
|                rand_Tensor_Torch |         rand |      65536 |     int32 |       cuda |               NA |                NA |             NA |     ? |       ? |       No |
|         rand_RawTensor_Reference |         rand |      65536 |     int32 |       cuda |               NA |                NA |             NA |     ? |       ? |       No |
|            rand_Tensor_Reference |         rand |      65536 |     int32 |       cuda |               NA |                NA |             NA |     ? |       ? |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                 addition_PyTorch |     addition |      65536 |     int32 |       cuda |      4,747.64 μs |      1,678.651 μs |      92.013 μs |  1.00 |    0.00 |      Yes |
|              addition_TorchSharp |     addition |      65536 |     int32 |       cuda |        593.70 μs |      1,099.835 μs |      60.286 μs |  0.13 |    0.01 |       No |
|         addition_RawTensor_Torch |     addition |      65536 |     int32 |       cuda |        726.67 μs |        178.008 μs |       9.757 μs |  0.15 |    0.00 |       No |
|            addition_Tensor_Torch |     addition |      65536 |     int32 |       cuda |        898.90 μs |        541.832 μs |      29.700 μs |  0.19 |    0.00 |       No |
|     addition_RawTensor_Reference |     addition |      65536 |     int32 |       cuda |      9,364.79 μs |      3,488.875 μs |     191.237 μs |  1.97 |    0.04 |       No |
|        addition_Tensor_Reference |     addition |      65536 |     int32 |       cuda |      9,968.83 μs |        458.006 μs |      25.105 μs |  2.10 |    0.04 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                addScalar_PyTorch |    addScalar |      65536 |     int32 |       cuda |      4,748.29 μs |        411.129 μs |      22.535 μs |  1.00 |    0.00 |      Yes |
|             addScalar_TorchSharp |    addScalar |      65536 |     int32 |       cuda |        748.63 μs |        115.878 μs |       6.352 μs |  0.16 |    0.00 |       No |
|        addScalar_RawTensor_Torch |    addScalar |      65536 |     int32 |       cuda |      7,338.93 μs |      7,379.614 μs |     404.502 μs |  1.55 |    0.09 |       No |
|           addScalar_Tensor_Torch |    addScalar |      65536 |     int32 |       cuda |      6,797.40 μs |     21,971.237 μs |   1,204.318 μs |  1.43 |    0.25 |       No |
|    addScalar_RawTensor_Reference |    addScalar |      65536 |     int32 |       cuda |               NA |                NA |             NA |     ? |       ? |       No |
|       addScalar_Tensor_Reference |    addScalar |      65536 |     int32 |       cuda |    122,328.23 μs |     81,151.487 μs |   4,448.187 μs | 25.76 |    1.01 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|             addWithAlpha_PyTorch | addWithAlpha |      65536 |     int32 |       cuda |      3,768.17 μs |         31.309 μs |       1.716 μs |  1.00 |    0.00 |      Yes |
|          addWithAlpha_TorchSharp | addWithAlpha |      65536 |     int32 |       cuda |        564.13 μs |        238.263 μs |      13.060 μs |  0.15 |    0.00 |       No |
|     addWithAlpha_RawTensor_Torch | addWithAlpha |      65536 |     int32 |       cuda |      1,533.97 μs |      1,242.812 μs |      68.123 μs |  0.41 |    0.02 |       No |
|        addWithAlpha_Tensor_Torch | addWithAlpha |      65536 |     int32 |       cuda |      4,381.63 μs |      6,884.086 μs |     377.340 μs |  1.16 |    0.10 |       No |
| addWithAlpha_RawTensor_Reference | addWithAlpha |      65536 |     int32 |       cuda |               NA |                NA |             NA |     ? |       ? |       No |
|    addWithAlpha_Tensor_Reference | addWithAlpha |      65536 |     int32 |       cuda |    237,306.47 μs |     20,311.537 μs |   1,113.344 μs | 62.98 |    0.31 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|               addInPlace_PyTorch |   addInPlace |      65536 |     int32 |       cuda |      3,776.74 μs |        423.485 μs |      23.213 μs |  1.00 |    0.00 |      Yes |
|            addInPlace_TorchSharp |   addInPlace |      65536 |     int32 |       cuda |        473.20 μs |      1,467.926 μs |      80.462 μs |  0.13 |    0.02 |       No |
|       addInPlace_RawTensor_Torch |   addInPlace |      65536 |     int32 |       cuda |        627.27 μs |        312.209 μs |      17.113 μs |  0.17 |    0.01 |       No |
|          addInPlace_Tensor_Torch |   addInPlace |      65536 |     int32 |       cuda |        972.10 μs |        239.966 μs |      13.153 μs |  0.26 |    0.00 |       No |
|   addInPlace_RawTensor_Reference |   addInPlace |      65536 |     int32 |       cuda |      9,170.49 μs |      3,356.861 μs |     184.001 μs |  2.43 |    0.06 |       No |
|      addInPlace_Tensor_Reference |   addInPlace |      65536 |     int32 |       cuda |     10,026.64 μs |        685.724 μs |      37.587 μs |  2.65 |    0.02 |       No |
|                                  |              |            |           |            |                  |                   |                |       |         |          |
|                   matmul_PyTorch |       matmul |      65536 |     int32 |       cuda |               NA |                NA |             NA |     ? |       ? |      Yes |
|                matmul_TorchSharp |       matmul |      65536 |     int32 |       cuda |               NA |                NA |             NA |     ? |       ? |       No |
|           matmul_RawTensor_Torch |       matmul |      65536 |     int32 |       cuda |        486.30 μs |        631.762 μs |      34.629 μs |     ? |       ? |       No |
|              matmul_Tensor_Torch |       matmul |      65536 |     int32 |       cuda |        598.70 μs |      3,330.118 μs |     182.535 μs |     ? |       ? |       No |
|       matmul_RawTensor_Reference |       matmul |      65536 |     int32 |       cuda |    975,517.27 μs |    194,262.253 μs |  10,648.170 μs |     ? |       ? |       No |
|          matmul_Tensor_Reference |       matmul |      65536 |     int32 |       cuda |    970,641.93 μs |    145,490.901 μs |   7,974.847 μs |     ? |       ? |       No |

Benchmarks with issues:
  BasicTensorOps.matmul_PyTorch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=16, dtypeName=float32, deviceName=cpu]
  BasicTensorOps.matmul_PyTorch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=16, dtypeName=float32, deviceName=cuda]
  BasicTensorOps.matmul_PyTorch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=16, dtypeName=float64, deviceName=cpu]
  BasicTensorOps.matmul_PyTorch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=16, dtypeName=float64, deviceName=cuda]
  BasicTensorOps.rand_PyTorch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=16, dtypeName=int32, deviceName=cpu]
  BasicTensorOps.rand_RawTensor_Torch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=16, dtypeName=int32, deviceName=cpu]
  BasicTensorOps.rand_Tensor_Torch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=16, dtypeName=int32, deviceName=cpu]
  BasicTensorOps.rand_RawTensor_Reference: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=16, dtypeName=int32, deviceName=cpu]
  BasicTensorOps.rand_Tensor_Reference: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=16, dtypeName=int32, deviceName=cpu]
  BasicTensorOps.addScalar_RawTensor_Reference: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=16, dtypeName=int32, deviceName=cpu]
  BasicTensorOps.addWithAlpha_RawTensor_Reference: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=16, dtypeName=int32, deviceName=cpu]
  BasicTensorOps.matmul_PyTorch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=16, dtypeName=int32, deviceName=cpu]
  BasicTensorOps.rand_PyTorch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=16, dtypeName=int32, deviceName=cuda]
  BasicTensorOps.rand_RawTensor_Torch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=16, dtypeName=int32, deviceName=cuda]
  BasicTensorOps.rand_Tensor_Torch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=16, dtypeName=int32, deviceName=cuda]
  BasicTensorOps.rand_RawTensor_Reference: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=16, dtypeName=int32, deviceName=cuda]
  BasicTensorOps.rand_Tensor_Reference: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=16, dtypeName=int32, deviceName=cuda]
  BasicTensorOps.addScalar_RawTensor_Reference: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=16, dtypeName=int32, deviceName=cuda]
  BasicTensorOps.addWithAlpha_RawTensor_Reference: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=16, dtypeName=int32, deviceName=cuda]
  BasicTensorOps.matmul_PyTorch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=16, dtypeName=int32, deviceName=cuda]
  BasicTensorOps.matmul_TorchSharp: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=16, dtypeName=int32, deviceName=cuda]
  BasicTensorOps.matmul_PyTorch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=2048, dtypeName=float32, deviceName=cpu]
  BasicTensorOps.matmul_PyTorch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=2048, dtypeName=float32, deviceName=cuda]
  BasicTensorOps.matmul_PyTorch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=2048, dtypeName=float64, deviceName=cpu]
  BasicTensorOps.matmul_PyTorch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=2048, dtypeName=float64, deviceName=cuda]
  BasicTensorOps.rand_PyTorch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=2048, dtypeName=int32, deviceName=cpu]
  BasicTensorOps.rand_RawTensor_Torch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=2048, dtypeName=int32, deviceName=cpu]
  BasicTensorOps.rand_Tensor_Torch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=2048, dtypeName=int32, deviceName=cpu]
  BasicTensorOps.rand_RawTensor_Reference: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=2048, dtypeName=int32, deviceName=cpu]
  BasicTensorOps.rand_Tensor_Reference: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=2048, dtypeName=int32, deviceName=cpu]
  BasicTensorOps.addScalar_RawTensor_Reference: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=2048, dtypeName=int32, deviceName=cpu]
  BasicTensorOps.addWithAlpha_RawTensor_Reference: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=2048, dtypeName=int32, deviceName=cpu]
  BasicTensorOps.matmul_PyTorch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=2048, dtypeName=int32, deviceName=cpu]
  BasicTensorOps.rand_PyTorch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=2048, dtypeName=int32, deviceName=cuda]
  BasicTensorOps.rand_RawTensor_Torch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=2048, dtypeName=int32, deviceName=cuda]
  BasicTensorOps.rand_Tensor_Torch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=2048, dtypeName=int32, deviceName=cuda]
  BasicTensorOps.rand_RawTensor_Reference: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=2048, dtypeName=int32, deviceName=cuda]
  BasicTensorOps.rand_Tensor_Reference: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=2048, dtypeName=int32, deviceName=cuda]
  BasicTensorOps.addScalar_RawTensor_Reference: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=2048, dtypeName=int32, deviceName=cuda]
  BasicTensorOps.addWithAlpha_RawTensor_Reference: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=2048, dtypeName=int32, deviceName=cuda]
  BasicTensorOps.matmul_PyTorch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=2048, dtypeName=int32, deviceName=cuda]
  BasicTensorOps.matmul_TorchSharp: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=2048, dtypeName=int32, deviceName=cuda]
  BasicTensorOps.matmul_PyTorch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=65536, dtypeName=float32, deviceName=cpu]
  BasicTensorOps.matmul_PyTorch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=65536, dtypeName=float32, deviceName=cuda]
  BasicTensorOps.matmul_PyTorch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=65536, dtypeName=float64, deviceName=cpu]
  BasicTensorOps.matmul_PyTorch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=65536, dtypeName=float64, deviceName=cuda]
  BasicTensorOps.rand_PyTorch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=65536, dtypeName=int32, deviceName=cpu]
  BasicTensorOps.rand_RawTensor_Torch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=65536, dtypeName=int32, deviceName=cpu]
  BasicTensorOps.rand_Tensor_Torch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=65536, dtypeName=int32, deviceName=cpu]
  BasicTensorOps.rand_RawTensor_Reference: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=65536, dtypeName=int32, deviceName=cpu]
  BasicTensorOps.rand_Tensor_Reference: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=65536, dtypeName=int32, deviceName=cpu]
  BasicTensorOps.addScalar_RawTensor_Reference: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=65536, dtypeName=int32, deviceName=cpu]
  BasicTensorOps.addWithAlpha_RawTensor_Reference: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=65536, dtypeName=int32, deviceName=cpu]
  BasicTensorOps.matmul_PyTorch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=65536, dtypeName=int32, deviceName=cpu]
  BasicTensorOps.rand_PyTorch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=65536, dtypeName=int32, deviceName=cuda]
  BasicTensorOps.rand_RawTensor_Torch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=65536, dtypeName=int32, deviceName=cuda]
  BasicTensorOps.rand_Tensor_Torch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=65536, dtypeName=int32, deviceName=cuda]
  BasicTensorOps.rand_RawTensor_Reference: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=65536, dtypeName=int32, deviceName=cuda]
  BasicTensorOps.rand_Tensor_Reference: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=65536, dtypeName=int32, deviceName=cuda]
  BasicTensorOps.addScalar_RawTensor_Reference: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=65536, dtypeName=int32, deviceName=cuda]
  BasicTensorOps.addWithAlpha_RawTensor_Reference: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=65536, dtypeName=int32, deviceName=cuda]
  BasicTensorOps.matmul_PyTorch: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=65536, dtypeName=int32, deviceName=cuda]
  BasicTensorOps.matmul_TorchSharp: ShortRun(IterationCount=3, LaunchCount=1, WarmupCount=3) [tensorSize=65536, dtypeName=int32, deviceName=cuda]