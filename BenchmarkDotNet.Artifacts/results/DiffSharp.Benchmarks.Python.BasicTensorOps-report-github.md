``` ini

BenchmarkDotNet=v0.12.1, OS=Windows 10.0.17134.1792 (1803/April2018Update/Redstone4)
Intel Xeon CPU E5-1620 0 3.60GHz, 1 CPU, 8 logical and 4 physical cores
.NET Core SDK=5.0.100
  [Host]     : .NET Core 3.1.9 (CoreCLR 4.700.20.47201, CoreFX 4.700.20.47203), X64 RyuJIT DEBUG
  DefaultJob : .NET Core 3.1.9 (CoreCLR 4.700.20.47201, CoreFX 4.700.20.47203), X64 RyuJIT


```
|               Method |   Categories | tensorSize | dtypeName | deviceName |         Mean |       Error |      StdDev |       Median | Baseline |
|--------------------- |------------- |----------- |---------- |----------- |-------------:|------------:|------------:|-------------:|--------- |
|  **fromCpuData_PyTorch** |  **fromCpuData** |         **16** |   **float32** |        **cpu** |   **528.251 ms** |   **7.1480 ms** |   **7.0203 ms** |   **528.263 ms** |       **No** |
|  **fromCpuData_PyTorch** |  **fromCpuData** |         **16** |   **float32** |       **cuda** | **3,459.289 ms** |  **68.1239 ms** | **108.0517 ms** | **3,423.512 ms** |       **No** |
|  **fromCpuData_PyTorch** |  **fromCpuData** |         **16** |   **float64** |        **cpu** |   **535.892 ms** |   **6.3468 ms** |   **5.9368 ms** |   **536.885 ms** |       **No** |
|  **fromCpuData_PyTorch** |  **fromCpuData** |         **16** |   **float64** |       **cuda** | **3,515.693 ms** |  **67.8496 ms** | **118.8331 ms** | **3,471.741 ms** |       **No** |
|  **fromCpuData_PyTorch** |  **fromCpuData** |         **16** |     **int32** |        **cpu** |   **504.074 ms** |   **9.7049 ms** |  **17.5000 ms** |   **497.700 ms** |       **No** |
|  **fromCpuData_PyTorch** |  **fromCpuData** |         **16** |     **int32** |       **cuda** | **3,502.036 ms** |  **69.3749 ms** | **130.3029 ms** | **3,435.864 ms** |       **No** |
|  **fromCpuData_PyTorch** |  **fromCpuData** |       **2048** |   **float32** |        **cpu** |    **29.163 ms** |   **0.5684 ms** |   **0.5582 ms** |    **29.037 ms** |       **No** |
|  **fromCpuData_PyTorch** |  **fromCpuData** |       **2048** |   **float32** |       **cuda** |    **52.264 ms** |   **0.5889 ms** |   **0.6546 ms** |    **52.211 ms** |       **No** |
|  **fromCpuData_PyTorch** |  **fromCpuData** |       **2048** |   **float64** |        **cpu** |    **29.226 ms** |   **0.4542 ms** |   **0.4249 ms** |    **29.350 ms** |       **No** |
|  **fromCpuData_PyTorch** |  **fromCpuData** |       **2048** |   **float64** |       **cuda** |    **57.338 ms** |   **1.6136 ms** |   **4.7578 ms** |    **54.610 ms** |       **No** |
|  **fromCpuData_PyTorch** |  **fromCpuData** |       **2048** |     **int32** |        **cpu** |    **22.228 ms** |   **0.2746 ms** |   **0.2434 ms** |    **22.168 ms** |       **No** |
|  **fromCpuData_PyTorch** |  **fromCpuData** |       **2048** |     **int32** |       **cuda** |    **45.845 ms** |   **0.4198 ms** |   **0.3927 ms** |    **45.844 ms** |       **No** |
|  **fromCpuData_PyTorch** |  **fromCpuData** |      **65536** |   **float32** |        **cpu** |    **30.638 ms** |   **0.3532 ms** |   **0.3131 ms** |    **30.523 ms** |       **No** |
|  **fromCpuData_PyTorch** |  **fromCpuData** |      **65536** |   **float32** |       **cuda** |    **32.691 ms** |   **0.3489 ms** |   **0.3093 ms** |    **32.793 ms** |       **No** |
|  **fromCpuData_PyTorch** |  **fromCpuData** |      **65536** |   **float64** |        **cpu** |    **30.983 ms** |   **0.3030 ms** |   **0.2686 ms** |    **31.042 ms** |       **No** |
|  **fromCpuData_PyTorch** |  **fromCpuData** |      **65536** |   **float64** |       **cuda** |    **32.538 ms** |   **0.2320 ms** |   **0.2056 ms** |    **32.518 ms** |       **No** |
|  **fromCpuData_PyTorch** |  **fromCpuData** |      **65536** |     **int32** |        **cpu** |    **24.413 ms** |   **0.4826 ms** |   **0.6275 ms** |    **24.289 ms** |       **No** |
|  **fromCpuData_PyTorch** |  **fromCpuData** |      **65536** |     **int32** |       **cuda** |    **25.957 ms** |   **0.3758 ms** |   **0.3332 ms** |    **26.033 ms** |       **No** |
|                      |              |            |           |            |              |             |             |              |          |
|        **zeros_PyTorch** |        **zeros** |         **16** |   **float32** |        **cpu** | **1,645.760 ms** |  **28.3134 ms** |  **35.8074 ms** | **1,632.159 ms** |       **No** |
|        **zeros_PyTorch** |        **zeros** |         **16** |   **float32** |       **cuda** | **5,047.589 ms** |  **53.2548 ms** |  **41.5779 ms** | **5,028.865 ms** |       **No** |
|        **zeros_PyTorch** |        **zeros** |         **16** |   **float64** |        **cpu** | **1,652.755 ms** |  **16.3041 ms** |  **12.7292 ms** | **1,652.044 ms** |       **No** |
|        **zeros_PyTorch** |        **zeros** |         **16** |   **float64** |       **cuda** | **5,145.732 ms** |  **97.4609 ms** | **112.2361 ms** | **5,124.099 ms** |       **No** |
|        **zeros_PyTorch** |        **zeros** |         **16** |     **int32** |        **cpu** | **1,616.289 ms** |  **14.5062 ms** |  **12.1133 ms** | **1,611.865 ms** |       **No** |
|        **zeros_PyTorch** |        **zeros** |         **16** |     **int32** |       **cuda** | **5,051.074 ms** |  **99.2790 ms** | **154.5655 ms** | **4,998.708 ms** |       **No** |
|        **zeros_PyTorch** |        **zeros** |       **2048** |   **float32** |        **cpu** |    **15.593 ms** |   **0.1425 ms** |   **0.1333 ms** |    **15.596 ms** |       **No** |
|        **zeros_PyTorch** |        **zeros** |       **2048** |   **float32** |       **cuda** |    **39.707 ms** |   **0.5796 ms** |   **0.6675 ms** |    **39.532 ms** |       **No** |
|        **zeros_PyTorch** |        **zeros** |       **2048** |   **float64** |        **cpu** |    **18.210 ms** |   **0.3358 ms** |   **0.5517 ms** |    **17.986 ms** |       **No** |
|        **zeros_PyTorch** |        **zeros** |       **2048** |   **float64** |       **cuda** |    **39.146 ms** |   **0.5106 ms** |   **0.4526 ms** |    **39.040 ms** |       **No** |
|        **zeros_PyTorch** |        **zeros** |       **2048** |     **int32** |        **cpu** |    **14.603 ms** |   **0.2878 ms** |   **0.3199 ms** |    **14.462 ms** |       **No** |
|        **zeros_PyTorch** |        **zeros** |       **2048** |     **int32** |       **cuda** |    **37.855 ms** |   **0.1490 ms** |   **0.1321 ms** |    **37.860 ms** |       **No** |
|        **zeros_PyTorch** |        **zeros** |      **65536** |   **float32** |        **cpu** |     **1.951 ms** |   **0.1071 ms** |   **0.3142 ms** |     **1.851 ms** |       **No** |
|        **zeros_PyTorch** |        **zeros** |      **65536** |   **float32** |       **cuda** |     **1.444 ms** |   **0.0287 ms** |   **0.0648 ms** |     **1.428 ms** |       **No** |
|        **zeros_PyTorch** |        **zeros** |      **65536** |   **float64** |        **cpu** |     **2.791 ms** |   **0.1252 ms** |   **0.3691 ms** |     **2.808 ms** |       **No** |
|        **zeros_PyTorch** |        **zeros** |      **65536** |   **float64** |       **cuda** |     **1.385 ms** |   **0.0256 ms** |   **0.0449 ms** |     **1.368 ms** |       **No** |
|        **zeros_PyTorch** |        **zeros** |      **65536** |     **int32** |        **cpu** |     **1.771 ms** |   **0.0676 ms** |   **0.1908 ms** |     **1.721 ms** |       **No** |
|        **zeros_PyTorch** |        **zeros** |      **65536** |     **int32** |       **cuda** |     **1.429 ms** |   **0.0284 ms** |   **0.0801 ms** |     **1.418 ms** |       **No** |
|                      |              |            |           |            |              |             |             |              |          |
|         **ones_PyTorch** |         **ones** |         **16** |   **float32** |        **cpu** | **1,669.590 ms** |  **29.0277 ms** |  **65.5203 ms** | **1,643.129 ms** |       **No** |
|         **ones_PyTorch** |         **ones** |         **16** |   **float32** |       **cuda** | **5,078.890 ms** | **101.1625 ms** | **197.3096 ms** | **4,997.626 ms** |       **No** |
|         **ones_PyTorch** |         **ones** |         **16** |   **float64** |        **cpu** | **1,659.131 ms** |  **17.2865 ms** |  **14.4351 ms** | **1,654.546 ms** |       **No** |
|         **ones_PyTorch** |         **ones** |         **16** |   **float64** |       **cuda** | **5,033.843 ms** |  **93.6214 ms** |  **82.9929 ms** | **5,010.243 ms** |       **No** |
|         **ones_PyTorch** |         **ones** |         **16** |     **int32** |        **cpu** | **1,597.712 ms** |  **24.0062 ms** |  **28.5776 ms** | **1,592.619 ms** |       **No** |
|         **ones_PyTorch** |         **ones** |         **16** |     **int32** |       **cuda** | **4,971.522 ms** |  **92.3508 ms** | **146.4781 ms** | **4,930.065 ms** |       **No** |
|         **ones_PyTorch** |         **ones** |       **2048** |   **float32** |        **cpu** |    **14.113 ms** |   **0.0806 ms** |   **0.0715 ms** |    **14.117 ms** |       **No** |
|         **ones_PyTorch** |         **ones** |       **2048** |   **float32** |       **cuda** |    **38.468 ms** |   **0.2790 ms** |   **0.2178 ms** |    **38.410 ms** |       **No** |
|         **ones_PyTorch** |         **ones** |       **2048** |   **float64** |        **cpu** |    **16.665 ms** |   **0.3153 ms** |   **0.2950 ms** |    **16.567 ms** |       **No** |
|         **ones_PyTorch** |         **ones** |       **2048** |   **float64** |       **cuda** |    **39.385 ms** |   **0.4292 ms** |   **0.3805 ms** |    **39.491 ms** |       **No** |
|         **ones_PyTorch** |         **ones** |       **2048** |     **int32** |        **cpu** |    **14.154 ms** |   **0.2742 ms** |   **0.2565 ms** |    **14.074 ms** |       **No** |
|         **ones_PyTorch** |         **ones** |       **2048** |     **int32** |       **cuda** |    **40.657 ms** |   **0.5858 ms** |   **0.5193 ms** |    **40.520 ms** |       **No** |
|         **ones_PyTorch** |         **ones** |      **65536** |   **float32** |        **cpu** |     **1.899 ms** |   **0.1003 ms** |   **0.2893 ms** |     **1.813 ms** |       **No** |
|         **ones_PyTorch** |         **ones** |      **65536** |   **float32** |       **cuda** |     **1.381 ms** |   **0.0276 ms** |   **0.0577 ms** |     **1.361 ms** |       **No** |
|         **ones_PyTorch** |         **ones** |      **65536** |   **float64** |        **cpu** |     **2.795 ms** |   **0.1454 ms** |   **0.4263 ms** |     **2.736 ms** |       **No** |
|         **ones_PyTorch** |         **ones** |      **65536** |   **float64** |       **cuda** |     **1.391 ms** |   **0.0276 ms** |   **0.0639 ms** |     **1.369 ms** |       **No** |
|         **ones_PyTorch** |         **ones** |      **65536** |     **int32** |        **cpu** |     **1.869 ms** |   **0.1266 ms** |   **0.3653 ms** |     **1.716 ms** |       **No** |
|         **ones_PyTorch** |         **ones** |      **65536** |     **int32** |       **cuda** |     **1.384 ms** |   **0.0275 ms** |   **0.0643 ms** |     **1.370 ms** |       **No** |
|                      |              |            |           |            |              |             |             |              |          |
|         **rand_PyTorch** |         **rand** |         **16** |   **float32** |        **cpu** | **1,981.077 ms** |  **39.2000 ms** |  **99.0635 ms** | **1,933.657 ms** |       **No** |
|         **rand_PyTorch** |         **rand** |         **16** |   **float32** |       **cuda** | **5,406.589 ms** |  **93.0025 ms** | **133.3814 ms** | **5,361.649 ms** |       **No** |
|         **rand_PyTorch** |         **rand** |         **16** |   **float64** |        **cpu** | **1,971.714 ms** |  **39.0232 ms** |  **68.3460 ms** | **1,937.530 ms** |       **No** |
|         **rand_PyTorch** |         **rand** |         **16** |   **float64** |       **cuda** | **5,364.707 ms** | **103.9160 ms** | **123.7046 ms** | **5,378.001 ms** |       **No** |
|         **rand_PyTorch** |         **rand** |         **16** |     **int32** |        **cpu** |           **NA** |          **NA** |          **NA** |           **NA** |       **No** |
|         **rand_PyTorch** |         **rand** |         **16** |     **int32** |       **cuda** |           **NA** |          **NA** |          **NA** |           **NA** |       **No** |
|         **rand_PyTorch** |         **rand** |       **2048** |   **float32** |        **cpu** |    **32.000 ms** |   **0.3424 ms** |   **0.3202 ms** |    **31.968 ms** |       **No** |
|         **rand_PyTorch** |         **rand** |       **2048** |   **float32** |       **cuda** |    **40.783 ms** |   **0.6180 ms** |   **0.5478 ms** |    **40.983 ms** |       **No** |
|         **rand_PyTorch** |         **rand** |       **2048** |   **float64** |        **cpu** |    **56.496 ms** |   **0.9911 ms** |   **0.9270 ms** |    **56.503 ms** |       **No** |
|         **rand_PyTorch** |         **rand** |       **2048** |   **float64** |       **cuda** |    **42.946 ms** |   **0.7812 ms** |   **0.6523 ms** |    **42.723 ms** |       **No** |
|         **rand_PyTorch** |         **rand** |       **2048** |     **int32** |        **cpu** |           **NA** |          **NA** |          **NA** |           **NA** |       **No** |
|         **rand_PyTorch** |         **rand** |       **2048** |     **int32** |       **cuda** |           **NA** |          **NA** |          **NA** |           **NA** |       **No** |
|         **rand_PyTorch** |         **rand** |      **65536** |   **float32** |        **cpu** |    **18.470 ms** |   **0.1712 ms** |   **0.1602 ms** |    **18.506 ms** |       **No** |
|         **rand_PyTorch** |         **rand** |      **65536** |   **float32** |       **cuda** |     **1.432 ms** |   **0.0277 ms** |   **0.0406 ms** |     **1.420 ms** |       **No** |
|         **rand_PyTorch** |         **rand** |      **65536** |   **float64** |        **cpu** |    **39.024 ms** |   **0.2519 ms** |   **0.2233 ms** |    **38.972 ms** |       **No** |
|         **rand_PyTorch** |         **rand** |      **65536** |   **float64** |       **cuda** |     **1.480 ms** |   **0.0280 ms** |   **0.0392 ms** |     **1.478 ms** |       **No** |
|         **rand_PyTorch** |         **rand** |      **65536** |     **int32** |        **cpu** |           **NA** |          **NA** |          **NA** |           **NA** |       **No** |
|         **rand_PyTorch** |         **rand** |      **65536** |     **int32** |       **cuda** |           **NA** |          **NA** |          **NA** |           **NA** |       **No** |
|                      |              |            |           |            |              |             |             |              |          |
|     **addition_PyTorch** |     **addition** |         **16** |   **float32** |        **cpu** |   **742.510 ms** |  **12.4315 ms** |  **11.6284 ms** |   **741.321 ms** |       **No** |
|     **addition_PyTorch** |     **addition** |         **16** |   **float32** |       **cuda** | **3,229.725 ms** |  **62.8968 ms** |  **69.9096 ms** | **3,209.733 ms** |       **No** |
|     **addition_PyTorch** |     **addition** |         **16** |   **float64** |        **cpu** |   **754.081 ms** |   **8.0423 ms** |   **7.5228 ms** |   **755.628 ms** |       **No** |
|     **addition_PyTorch** |     **addition** |         **16** |   **float64** |       **cuda** | **3,145.584 ms** |  **60.0503 ms** |  **61.6672 ms** | **3,121.837 ms** |       **No** |
|     **addition_PyTorch** |     **addition** |         **16** |     **int32** |        **cpu** |   **714.067 ms** |   **5.5989 ms** |   **5.2372 ms** |   **714.155 ms** |       **No** |
|     **addition_PyTorch** |     **addition** |         **16** |     **int32** |       **cuda** | **3,260.667 ms** |  **60.6064 ms** |  **78.8055 ms** | **3,231.562 ms** |       **No** |
|     **addition_PyTorch** |     **addition** |       **2048** |   **float32** |        **cpu** |     **9.307 ms** |   **0.1852 ms** |   **0.4879 ms** |     **9.193 ms** |       **No** |
|     **addition_PyTorch** |     **addition** |       **2048** |   **float32** |       **cuda** |    **25.150 ms** |   **0.4066 ms** |   **0.3994 ms** |    **25.102 ms** |       **No** |
|     **addition_PyTorch** |     **addition** |       **2048** |   **float64** |        **cpu** |    **12.750 ms** |   **0.2497 ms** |   **0.4438 ms** |    **12.780 ms** |       **No** |
|     **addition_PyTorch** |     **addition** |       **2048** |   **float64** |       **cuda** |    **24.604 ms** |   **0.2859 ms** |   **0.2674 ms** |    **24.567 ms** |       **No** |
|     **addition_PyTorch** |     **addition** |       **2048** |     **int32** |        **cpu** |     **9.266 ms** |   **0.1837 ms** |   **0.3451 ms** |     **9.123 ms** |       **No** |
|     **addition_PyTorch** |     **addition** |       **2048** |     **int32** |       **cuda** |    **25.396 ms** |   **0.7400 ms** |   **2.1701 ms** |    **24.280 ms** |       **No** |
|     **addition_PyTorch** |     **addition** |      **65536** |   **float32** |        **cpu** |     **7.553 ms** |   **0.3445 ms** |   **0.9429 ms** |     **7.462 ms** |       **No** |
|     **addition_PyTorch** |     **addition** |      **65536** |   **float32** |       **cuda** |     **5.195 ms** |   **0.1026 ms** |   **0.1714 ms** |     **5.199 ms** |       **No** |
|     **addition_PyTorch** |     **addition** |      **65536** |   **float64** |        **cpu** |    **10.074 ms** |   **0.8095 ms** |   **2.3868 ms** |     **9.759 ms** |       **No** |
|     **addition_PyTorch** |     **addition** |      **65536** |   **float64** |       **cuda** |     **5.403 ms** |   **0.1024 ms** |   **0.2289 ms** |     **5.406 ms** |       **No** |
|     **addition_PyTorch** |     **addition** |      **65536** |     **int32** |        **cpu** |     **6.305 ms** |   **0.3180 ms** |   **0.8866 ms** |     **6.300 ms** |       **No** |
|     **addition_PyTorch** |     **addition** |      **65536** |     **int32** |       **cuda** |     **4.393 ms** |   **0.1014 ms** |   **0.2908 ms** |     **4.347 ms** |       **No** |
|                      |              |            |           |            |              |             |             |              |          |
|   **addInPlace_PyTorch** |   **addInPlace** |         **16** |   **float32** |        **cpu** |   **409.598 ms** |   **2.2956 ms** |   **1.9169 ms** |   **409.986 ms** |       **No** |
|   **addInPlace_PyTorch** |   **addInPlace** |         **16** |   **float32** |       **cuda** | **1,643.912 ms** |  **13.8597 ms** |  **11.5735 ms** | **1,639.398 ms** |       **No** |
|   **addInPlace_PyTorch** |   **addInPlace** |         **16** |   **float64** |        **cpu** |   **385.327 ms** |   **4.4139 ms** |   **3.6858 ms** |   **385.888 ms** |       **No** |
|   **addInPlace_PyTorch** |   **addInPlace** |         **16** |   **float64** |       **cuda** | **1,600.520 ms** |  **19.7346 ms** |  **16.4793 ms** | **1,599.511 ms** |       **No** |
|   **addInPlace_PyTorch** |   **addInPlace** |         **16** |     **int32** |        **cpu** |   **383.271 ms** |   **4.0174 ms** |   **3.5613 ms** |   **382.039 ms** |       **No** |
|   **addInPlace_PyTorch** |   **addInPlace** |         **16** |     **int32** |       **cuda** | **1,613.522 ms** |  **31.8994 ms** |  **32.7583 ms** | **1,602.715 ms** |       **No** |
|   **addInPlace_PyTorch** |   **addInPlace** |       **2048** |   **float32** |        **cpu** |     **5.516 ms** |   **0.1036 ms** |   **0.0865 ms** |     **5.493 ms** |       **No** |
|   **addInPlace_PyTorch** |   **addInPlace** |       **2048** |   **float32** |       **cuda** |    **13.091 ms** |   **0.2013 ms** |   **0.1784 ms** |    **13.081 ms** |       **No** |
|   **addInPlace_PyTorch** |   **addInPlace** |       **2048** |   **float64** |        **cpu** |     **7.582 ms** |   **0.1511 ms** |   **0.1965 ms** |     **7.507 ms** |       **No** |
|   **addInPlace_PyTorch** |   **addInPlace** |       **2048** |   **float64** |       **cuda** |    **13.340 ms** |   **0.2572 ms** |   **0.3434 ms** |    **13.321 ms** |       **No** |
|   **addInPlace_PyTorch** |   **addInPlace** |       **2048** |     **int32** |        **cpu** |     **5.523 ms** |   **0.0779 ms** |   **0.0650 ms** |     **5.527 ms** |       **No** |
|   **addInPlace_PyTorch** |   **addInPlace** |       **2048** |     **int32** |       **cuda** |    **12.789 ms** |   **0.2411 ms** |   **0.2476 ms** |    **12.714 ms** |       **No** |
|   **addInPlace_PyTorch** |   **addInPlace** |      **65536** |   **float32** |        **cpu** |     **6.895 ms** |   **0.3885 ms** |   **1.0635 ms** |     **6.739 ms** |       **No** |
|   **addInPlace_PyTorch** |   **addInPlace** |      **65536** |   **float32** |       **cuda** |     **4.988 ms** |   **0.0990 ms** |   **0.2676 ms** |     **4.922 ms** |       **No** |
|   **addInPlace_PyTorch** |   **addInPlace** |      **65536** |   **float64** |        **cpu** |     **8.604 ms** |   **0.4917 ms** |   **1.3789 ms** |     **8.934 ms** |       **No** |
|   **addInPlace_PyTorch** |   **addInPlace** |      **65536** |   **float64** |       **cuda** |     **4.887 ms** |   **0.0962 ms** |   **0.1710 ms** |     **4.843 ms** |       **No** |
|   **addInPlace_PyTorch** |   **addInPlace** |      **65536** |     **int32** |        **cpu** |     **5.706 ms** |   **0.3159 ms** |   **0.8807 ms** |     **5.769 ms** |       **No** |
|   **addInPlace_PyTorch** |   **addInPlace** |      **65536** |     **int32** |       **cuda** |     **3.919 ms** |   **0.0770 ms** |   **0.1502 ms** |     **3.929 ms** |       **No** |
|                      |              |            |           |            |              |             |             |              |          |
| **addWithAlpha_PyTorch** | **addWithAlpha** |         **16** |   **float32** |        **cpu** |   **530.222 ms** |   **3.0549 ms** |   **2.5510 ms** |   **530.745 ms** |       **No** |
| **addWithAlpha_PyTorch** | **addWithAlpha** |         **16** |   **float32** |       **cuda** | **1,873.067 ms** |  **28.0883 ms** |  **23.4550 ms** | **1,869.784 ms** |       **No** |
| **addWithAlpha_PyTorch** | **addWithAlpha** |         **16** |   **float64** |        **cpu** |   **482.067 ms** |   **6.8547 ms** |   **6.0765 ms** |   **480.838 ms** |       **No** |
| **addWithAlpha_PyTorch** | **addWithAlpha** |         **16** |   **float64** |       **cuda** | **1,782.789 ms** |  **15.0738 ms** |  **14.1000 ms** | **1,781.665 ms** |       **No** |
| **addWithAlpha_PyTorch** | **addWithAlpha** |         **16** |     **int32** |        **cpu** |   **485.122 ms** |   **8.1637 ms** |   **7.6364 ms** |   **484.073 ms** |       **No** |
| **addWithAlpha_PyTorch** | **addWithAlpha** |         **16** |     **int32** |       **cuda** | **1,804.856 ms** |  **17.7728 ms** |  **14.8411 ms** | **1,804.852 ms** |       **No** |
| **addWithAlpha_PyTorch** | **addWithAlpha** |       **2048** |   **float32** |        **cpu** |     **6.270 ms** |   **0.0828 ms** |   **0.0691 ms** |     **6.259 ms** |       **No** |
| **addWithAlpha_PyTorch** | **addWithAlpha** |       **2048** |   **float32** |       **cuda** |    **14.842 ms** |   **0.1932 ms** |   **0.1807 ms** |    **14.848 ms** |       **No** |
| **addWithAlpha_PyTorch** | **addWithAlpha** |       **2048** |   **float64** |        **cpu** |     **8.908 ms** |   **0.1770 ms** |   **0.3146 ms** |     **8.995 ms** |       **No** |
| **addWithAlpha_PyTorch** | **addWithAlpha** |       **2048** |   **float64** |       **cuda** |    **14.301 ms** |   **0.2444 ms** |   **0.2286 ms** |    **14.250 ms** |       **No** |
| **addWithAlpha_PyTorch** | **addWithAlpha** |       **2048** |     **int32** |        **cpu** |     **6.392 ms** |   **0.0855 ms** |   **0.0985 ms** |     **6.381 ms** |       **No** |
| **addWithAlpha_PyTorch** | **addWithAlpha** |       **2048** |     **int32** |       **cuda** |    **17.897 ms** |   **0.2732 ms** |   **0.2555 ms** |    **17.899 ms** |       **No** |
| **addWithAlpha_PyTorch** | **addWithAlpha** |      **65536** |   **float32** |        **cpu** |     **6.972 ms** |   **0.3282 ms** |   **0.8872 ms** |     **6.907 ms** |       **No** |
| **addWithAlpha_PyTorch** | **addWithAlpha** |      **65536** |   **float32** |       **cuda** |     **4.879 ms** |   **0.0974 ms** |   **0.1968 ms** |     **4.873 ms** |       **No** |
| **addWithAlpha_PyTorch** | **addWithAlpha** |      **65536** |   **float64** |        **cpu** |     **9.230 ms** |   **0.7637 ms** |   **2.2398 ms** |     **9.007 ms** |       **No** |
| **addWithAlpha_PyTorch** | **addWithAlpha** |      **65536** |   **float64** |       **cuda** |     **4.933 ms** |   **0.0972 ms** |   **0.1872 ms** |     **4.913 ms** |       **No** |
| **addWithAlpha_PyTorch** | **addWithAlpha** |      **65536** |     **int32** |        **cpu** |     **5.541 ms** |   **0.2620 ms** |   **0.7347 ms** |     **5.558 ms** |       **No** |
| **addWithAlpha_PyTorch** | **addWithAlpha** |      **65536** |     **int32** |       **cuda** |     **3.981 ms** |   **0.0790 ms** |   **0.2190 ms** |     **3.939 ms** |       **No** |
|                      |              |            |           |            |              |             |             |              |          |
|    **addScalar_PyTorch** |    **addScalar** |         **16** |   **float32** |        **cpu** | **1,938.403 ms** |  **16.2285 ms** |  **15.1802 ms** | **1,939.982 ms** |       **No** |
|    **addScalar_PyTorch** |    **addScalar** |         **16** |   **float32** |       **cuda** | **4,212.287 ms** |  **82.0461 ms** | **132.4891 ms** | **4,179.452 ms** |       **No** |
|    **addScalar_PyTorch** |    **addScalar** |         **16** |   **float64** |        **cpu** | **1,966.137 ms** |  **39.1755 ms** |  **49.5445 ms** | **1,949.195 ms** |       **No** |
|    **addScalar_PyTorch** |    **addScalar** |         **16** |   **float64** |       **cuda** | **4,186.714 ms** |  **81.4232 ms** |  **87.1219 ms** | **4,162.382 ms** |       **No** |
|    **addScalar_PyTorch** |    **addScalar** |         **16** |     **int32** |        **cpu** | **1,863.798 ms** |  **30.4151 ms** |  **28.4503 ms** | **1,855.691 ms** |       **No** |
|    **addScalar_PyTorch** |    **addScalar** |         **16** |     **int32** |       **cuda** | **4,333.180 ms** |  **82.6515 ms** | **107.4702 ms** | **4,285.018 ms** |       **No** |
|    **addScalar_PyTorch** |    **addScalar** |       **2048** |   **float32** |        **cpu** |    **19.365 ms** |   **0.2832 ms** |   **0.2649 ms** |    **19.361 ms** |       **No** |
|    **addScalar_PyTorch** |    **addScalar** |       **2048** |   **float32** |       **cuda** |    **33.306 ms** |   **0.3753 ms** |   **0.3511 ms** |    **33.394 ms** |       **No** |
|    **addScalar_PyTorch** |    **addScalar** |       **2048** |   **float64** |        **cpu** |    **23.412 ms** |   **0.2590 ms** |   **0.2296 ms** |    **23.387 ms** |       **No** |
|    **addScalar_PyTorch** |    **addScalar** |       **2048** |   **float64** |       **cuda** |    **32.836 ms** |   **0.1855 ms** |   **0.1735 ms** |    **32.841 ms** |       **No** |
|    **addScalar_PyTorch** |    **addScalar** |       **2048** |     **int32** |        **cpu** |    **18.673 ms** |   **0.2286 ms** |   **0.2026 ms** |    **18.639 ms** |       **No** |
|    **addScalar_PyTorch** |    **addScalar** |       **2048** |     **int32** |       **cuda** |    **32.789 ms** |   **0.2587 ms** |   **0.2420 ms** |    **32.731 ms** |       **No** |
|    **addScalar_PyTorch** |    **addScalar** |      **65536** |   **float32** |        **cpu** |     **7.565 ms** |   **0.3478 ms** |   **0.9810 ms** |     **7.455 ms** |       **No** |
|    **addScalar_PyTorch** |    **addScalar** |      **65536** |   **float32** |       **cuda** |     **5.397 ms** |   **0.1072 ms** |   **0.1792 ms** |     **5.389 ms** |       **No** |
|    **addScalar_PyTorch** |    **addScalar** |      **65536** |   **float64** |        **cpu** |    **10.924 ms** |   **0.8420 ms** |   **2.4695 ms** |    **10.788 ms** |       **No** |
|    **addScalar_PyTorch** |    **addScalar** |      **65536** |   **float64** |       **cuda** |     **5.528 ms** |   **0.1083 ms** |   **0.1809 ms** |     **5.527 ms** |       **No** |
|    **addScalar_PyTorch** |    **addScalar** |      **65536** |     **int32** |        **cpu** |     **7.417 ms** |   **0.5242 ms** |   **1.5292 ms** |     **7.018 ms** |       **No** |
|    **addScalar_PyTorch** |    **addScalar** |      **65536** |     **int32** |       **cuda** |     **4.664 ms** |   **0.0932 ms** |   **0.2520 ms** |     **4.627 ms** |       **No** |

Benchmarks with issues:
  BasicTensorOps.rand_PyTorch: DefaultJob [tensorSize=16, dtypeName=int32, deviceName=cpu]
  BasicTensorOps.rand_PyTorch: DefaultJob [tensorSize=16, dtypeName=int32, deviceName=cuda]
  BasicTensorOps.rand_PyTorch: DefaultJob [tensorSize=2048, dtypeName=int32, deviceName=cpu]
  BasicTensorOps.rand_PyTorch: DefaultJob [tensorSize=2048, dtypeName=int32, deviceName=cuda]
  BasicTensorOps.rand_PyTorch: DefaultJob [tensorSize=65536, dtypeName=int32, deviceName=cpu]
  BasicTensorOps.rand_PyTorch: DefaultJob [tensorSize=65536, dtypeName=int32, deviceName=cuda]