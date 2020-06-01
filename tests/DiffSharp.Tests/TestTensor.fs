namespace Tests

open NUnit.Framework
open DiffSharp
open System

[<TestFixture>]
type TestTensor () =
    [<SetUp>]
    member _.Setup () =
        ()

    member _.TestTensorCreateAllTensorTypesGeneric (ofDouble: double -> 'T) =
      // Test creating these types of tensors
      for combo in Combos.All do 
        let t0 = combo.tensor(ofDouble 1.)
        let t0ShapeCorrect = [||]
        let t0DimCorrect = 0

        Assert.AreEqual(t0ShapeCorrect, t0.shape)
        Assert.AreEqual(t0DimCorrect, t0.dim)
        Assert.AreEqual(combo.dtype, t0.dtype)

        let t1 = combo.tensor([ofDouble 1.; ofDouble 2.; ofDouble 3.])
        let t1ShapeCorrect = [|3|]
        let t1DimCorrect = 1

        Assert.AreEqual(t1ShapeCorrect, t1.shape)
        Assert.AreEqual(t1DimCorrect, t1.dim)
        Assert.AreEqual(combo.dtype, t1.dtype)

        let t2 = combo.tensor([[ofDouble 1.; ofDouble 2.; ofDouble 3.]; [ofDouble 4.; ofDouble 5.; ofDouble 6.]])
        let t2ShapeCorrect = [|2; 3|]
        let t2DimCorrect = 2
        Assert.AreEqual(t2ShapeCorrect, t2.shape)
        Assert.AreEqual(t2DimCorrect, t2.dim)
        Assert.AreEqual(combo.dtype, t2.dtype)

        let t3 = combo.tensor([[[ofDouble 1.; ofDouble 2.; ofDouble 3.]; [ofDouble 4.; ofDouble 5.; ofDouble 6.]]])
        let t3ShapeCorrect = [|1; 2; 3|]
        let t3DimCorrect = 3

        Assert.AreEqual(t3ShapeCorrect, t3.shape)
        Assert.AreEqual(t3DimCorrect, t3.dim)
        Assert.AreEqual(combo.dtype, t3.dtype)

        let t4 = combo.tensor([[[[ofDouble 1.; ofDouble 2.]]]])
        let t4ShapeCorrect = [|1; 1; 1; 2|]
        let t4DimCorrect = 4

        Assert.AreEqual(t4ShapeCorrect, t4.shape)
        Assert.AreEqual(t4DimCorrect, t4.dim)
        Assert.AreEqual(combo.dtype, t4.dtype)

    [<Test>]
    member this.TestTensorCreateAllTensorTypesFromFloat64Data() =
        this.TestTensorCreateAllTensorTypesGeneric id

    [<Test>]
    member this.TestTensorCreateAllTensorTypesFromFloat32Data() =
        this.TestTensorCreateAllTensorTypesGeneric float32

    [<Test>]
    member this.TestTensorCreateAllTensorTypesFromInt32Data() =
        this.TestTensorCreateAllTensorTypesGeneric int32

    [<Test>]
    member this.TestTensorCreateAllTensorTypesFromInt8Data() =
        this.TestTensorCreateAllTensorTypesGeneric int8

    [<Test>]
    member this.TestTensorCreateAllTensorTypesFromInt16Data() =
        this.TestTensorCreateAllTensorTypesGeneric int16

    [<Test>]
    member this.TestTensorCreateAllTensorTypesFromInt64Data() =
        this.TestTensorCreateAllTensorTypesGeneric int64

    [<Test>]
    member this.TestTensorCreateAllTensorTypesFromBoolData() =
        this.TestTensorCreateAllTensorTypesGeneric (fun i -> abs i >= 1.0)

        let t1 = dsharp.tensor([true, true])
        Assert.AreEqual(DType.Bool, t1.dtype)

        let t2 = dsharp.tensor([true, false])
        Assert.AreEqual(DType.Bool, t2.dtype)

        let t3 = dsharp.tensor([true; false])
        Assert.AreEqual(DType.Bool, t3.dtype)

        let t4 = dsharp.tensor([true; false], dtype=DType.Float32)
        Assert.AreEqual(DType.Float32, t4.dtype)

    [<Test>]
    member _.TestTensorCreate0 () =
      for combo in Combos.AllDevicesAndBackends do
        let t0 = combo.tensor(1.)
        let t0Shape = t0.shape
        let t0Dim = t0.dim
        let t0ShapeCorrect = [||]
        let t0DimCorrect = 0

        Assert.AreEqual(t0DimCorrect, t0Dim)
        Assert.AreEqual(t0ShapeCorrect, t0Shape)

    [<Test>]
    member _.TestTensorCreate1 () =
      for combo in Combos.AllDevicesAndBackends do
        // create from double list
        let t1 = combo.tensor([1.; 2.; 3.])
        let t1ShapeCorrect = [|3|]
        let t1DimCorrect = 1

        Assert.AreEqual(t1ShapeCorrect, t1.shape)
        Assert.AreEqual(t1DimCorrect, t1.dim)

        // create from double[]
        let t1Array = combo.tensor([| 1.; 2.; 3. |])

        Assert.AreEqual(t1ShapeCorrect, t1Array.shape)
        Assert.AreEqual(t1DimCorrect, t1Array.dim)

        // create from seq<double>
        let t1Seq = combo.tensor(seq { 1.; 2.; 3. })

        Assert.AreEqual(t1ShapeCorrect, t1Seq.shape)
        Assert.AreEqual(t1DimCorrect, t1Seq.dim)

    [<Test>]
    member _.TestTensorCreate2 () =
      for combo in Combos.AllDevicesAndBackends do
        let t2Values = [[1.; 2.; 3.]; [4.; 5.; 6.]]
        let t2ShapeCorrect = [|2; 3|]
        let t2DimCorrect = 2
        // let t2DTypeCorrect = DType.Float32
        let t2ValuesCorrect = array2D (List.map (List.map float32) t2Values)

        // create from double list list
        let t2 = combo.tensor([[1.; 2.; 3.]; [4.; 5.; 6.]])
        Assert.AreEqual(t2ShapeCorrect, t2.shape)
        Assert.AreEqual(t2DimCorrect, t2.dim)
        Assert.AreEqual(t2ValuesCorrect, t2.toArray())

        // create from double array list
        let t2ArrayList = combo.tensor([[|1.; 2.; 3.|]; [|4.; 5.; 6.|]])
        Assert.AreEqual(t2ShapeCorrect, t2ArrayList.shape)
        Assert.AreEqual(t2DimCorrect, t2ArrayList.dim)
        Assert.AreEqual(t2ValuesCorrect, t2ArrayList.toArray())

        // create from double list array
        let t2ListArray = combo.tensor([| [1.; 2.; 3.]; [4.; 5.; 6.] |])
        Assert.AreEqual(t2ShapeCorrect, t2ListArray.shape)
        Assert.AreEqual(t2DimCorrect, t2ListArray.dim)
        Assert.AreEqual(t2ValuesCorrect, t2ListArray.toArray())

        // create from double[][]
        let t2ArrayArray = combo.tensor([| [| 1.; 2.; 3. |]; [| 4.; 5.; 6.|] |])
        Assert.AreEqual(t2ShapeCorrect, t2ArrayArray.shape)
        Assert.AreEqual(t2DimCorrect, t2ArrayArray.dim)
        Assert.AreEqual(t2ValuesCorrect, t2ArrayArray.toArray())

        // create from double[,]
        let t2Array2D = combo.tensor(array2D [| [| 1.; 2.; 3. |]; [| 4.; 5.; 6.|] |])
        Assert.AreEqual(t2ShapeCorrect, t2Array2D.shape)
        Assert.AreEqual(t2DimCorrect, t2Array2D.dim)
        Assert.AreEqual(t2ValuesCorrect, t2Array2D.toArray())

        // create from seq<double[]>
        let t2ArraySeq = combo.tensor(seq { yield [| 1.; 2.; 3. |]; yield [| 4.; 5.; 6.|] })
        Assert.AreEqual(t2ShapeCorrect, t2ArraySeq.shape)
        Assert.AreEqual(t2DimCorrect, t2ArraySeq.dim)
        Assert.AreEqual(t2ValuesCorrect, t2ArraySeq.toArray())

        // create from seq<seq<double>>
        let t2SeqSeq = combo.tensor(seq { seq { 1.; 2.; 3. }; seq { 4.; 5.; 6.} })
        Assert.AreEqual(t2ShapeCorrect, t2SeqSeq.shape)
        Assert.AreEqual(t2DimCorrect, t2SeqSeq.dim)
        Assert.AreEqual(t2ValuesCorrect, t2SeqSeq.toArray())

        // create from (double * double * double) list list
        let t2TupleListList = combo.tensor([ [ 1., 2., 3. ]; [ 4., 5., 6. ] ])
        Assert.AreEqual(t2ShapeCorrect, t2TupleListList.shape)
        Assert.AreEqual(t2DimCorrect, t2TupleListList.dim)
        Assert.AreEqual(t2ValuesCorrect, t2TupleListList.toArray())

        // create from ((double * double * double) list * (double * double * double) list) list
        let t2TupleListTupleList = combo.tensor([ [ 1., 2., 3. ], [ 4., 5., 6. ] ])
        Assert.AreEqual(t2ShapeCorrect, t2TupleListTupleList.shape)
        Assert.AreEqual(t2DimCorrect, t2TupleListTupleList.dim)
        Assert.AreEqual(t2ValuesCorrect, t2TupleListTupleList.toArray())

        // create from (double * double * double)[]
        let t2TupleArray = combo.tensor([| [ 1., 2., 3. ]; [ 4., 5., 6. ] |])
        Assert.AreEqual(t2ShapeCorrect, t2TupleArray.shape)
        Assert.AreEqual(t2DimCorrect, t2TupleArray.dim)
        Assert.AreEqual(t2ValuesCorrect, t2TupleArray.toArray())

        // create from ((double * double * double) [] * (double * double * double) []) []
        let t2TupleArrayTupleArray = combo.tensor([| [| 1., 2., 3. |], [| 4., 5., 6. |] |])
        Assert.AreEqual(t2ShapeCorrect, t2TupleArrayTupleArray.shape)
        Assert.AreEqual(t2DimCorrect, t2TupleArrayTupleArray.dim)
        Assert.AreEqual(t2ValuesCorrect, t2TupleArrayTupleArray.toArray())
        Assert.AreEqual(t2ValuesCorrect, t2TupleArrayTupleArray.toArray())

        // create from (double * double * double)seq
        let t2TupleArray = combo.tensor(seq { [ 1., 2., 3. ]; [ 4., 5., 6. ] })
        Assert.AreEqual(t2ShapeCorrect, t2TupleArray.shape)
        Assert.AreEqual(t2DimCorrect, t2TupleArray.dim)
        Assert.AreEqual(t2ValuesCorrect, t2TupleArray.toArray())

        let t2TupleOfList = combo.tensor [[2.], [3.], [4.]]
        Assert.AreEqual([| 3; 1 |], t2TupleOfList.shape)
        Assert.AreEqual(array2D [ [2]; [3]; [4] ], t2TupleOfList.toArray())

    [<Test>]
    member _.TestTensorCreate3 () =
      for combo in Combos.AllDevicesAndBackends do
        let t3Values = [[[1.; 2.; 3.]; [4.; 5.; 6.]]]
        let t3 = combo.tensor(t3Values)
        let t3ShapeCorrect = [|1; 2; 3|]
        let t3DimCorrect = 3
        let t3ValuesCorrect = Util.array3D (List.map (List.map (List.map float32)) t3Values)

        Assert.AreEqual(t3ShapeCorrect, t3.shape)
        Assert.AreEqual(t3DimCorrect, t3.dim)
        Assert.AreEqual(t3ValuesCorrect, t3.toArray())

    [<Test>]
    member _.TestTensorCreate4 () =
      for combo in Combos.AllDevicesAndBackends do
        let t4Values = [[[[1.; 2.]]]]
        let t4 = combo.tensor(t4Values)
        let t4ShapeCorrect = [|1; 1; 1; 2|]
        let t4DimCorrect = 4
        let t4ValuesCorrect = Util.array4D (List.map (List.map (List.map (List.map float32))) t4Values)

        Assert.AreEqual(t4ShapeCorrect, t4.shape)
        Assert.AreEqual(t4DimCorrect, t4.dim)
        Assert.AreEqual(t4ValuesCorrect, t4.toArray())

    [<Test>]
    member _.TestTensorToArray () =
        for combo in Combos.All do 
            let a = array2D [[1.; 2.]; [3.; 4.]]
            let t = combo.tensor(a)
            let tToArrayCorrect = combo.arrayCreator2D a
            Assert.AreEqual(tToArrayCorrect, t.toArray())

    [<Test>]
    member _.TestTensorSaveLoad () =
        let fileName = System.IO.Path.GetTempFileName()
        for combo in Combos.All do 
            let a = combo.tensor([[1,2],[3,4]])
            a.save(fileName)
            let b = Tensor.load(fileName)
            Assert.AreEqual(a, b)

    [<Test>]
    member _.TestTensorClone () =
        for combo in Combos.All do 
            let a = combo.randint(0,100,[10;10])
            let b = a.clone()
            Assert.AreEqual(a, b)
            Assert.AreEqual(a.dtype, b.dtype)

    [<Test>]
    member _.TestTensorFull () =
        for combo in Combos.IntegralAndFloatingPoint do 
            let t1a = combo.full([2;3], 2.5)
            let t1b = combo.ones([2;3]) * 2.5
            let t2a = combo.full([], 2.5)
            let t2b = combo.ones([]) * 2.5
            Assert.AreEqual(t1a, t1b)
            Assert.AreEqual(t2a, t2b)

        for combo in Combos.All do 
            let t1 = combo.full([2], 1)
            let t1Expected = combo.tensor([1,1])
            Assert.AreEqual(t1, t1Expected)

    [<Test>]
    member _.TestTensorZero () =
        for combo in Combos.All do 
            let t1 = combo.zero()
            let t1Expected = combo.tensor(0)
            Assert.AreEqual(t1, t1Expected)
            Assert.AreEqual(t1.shape, ([| |]: int32[]) )
            Assert.AreEqual(t1.dtype, combo.dtype)

    [<Test>]
    member _.TestTensorZeros () =
        for combo in Combos.All do 
            let t0 = combo.zeros([])
            let t0Expected = combo.tensor(0)
            Assert.AreEqual(t0.shape, ([| |]: int32[]) )
            Assert.AreEqual(t0.dtype, combo.dtype)
            Assert.AreEqual(t0, t0Expected)

            let t1 = combo.zeros([2])
            let t1Expected = combo.tensor([0,0])
            Assert.AreEqual(t1.shape, ([| 2 |]: int32[]) )
            Assert.AreEqual(t1.dtype, combo.dtype)
            Assert.AreEqual(t1, t1Expected)

    [<Test>]
    member _.TestTensorOne () =
        for combo in Combos.All do 
            let t1 = combo.one()
            let t1Expected = combo.tensor(1)
            Assert.AreEqual(t1, t1Expected)
            Assert.AreEqual(t1.dtype, combo.dtype)
            Assert.AreEqual(t1.shape, ([| |]: int32[]) )

    [<Test>]
    member _.TestTensorOnes () =
        for combo in Combos.All do 
            let t0 = combo.ones([])
            let t0Expected = combo.tensor(1)
            Assert.AreEqual(t0.shape, ([| |]: int32[]) )
            Assert.AreEqual(t0.dtype, combo.dtype)
            Assert.AreEqual(t0, t0Expected)

            let t1 = combo.ones([2])
            let t1Expected = combo.tensor([1,1])
            Assert.AreEqual(t1, t1Expected)
    [<Test>]
    member _.TestTensorIsTensor () =
        for combo in Combos.All do 
            let a = 2.
            let b = combo.tensor(2.)
            Assert.True(not (dsharp.isTensor(a)))
            Assert.True(dsharp.isTensor(b))

    [<Test>]
    member _.TestTensorConvert () =
        for combo in Combos.IntegralAndFloatingPoint do
            let v = 2.
            let t = combo.tensor(v)
            let tsingle = single t
            let tdouble = double t
            let tint16 = int16 t
            let tint32 = int32 t
            let tint64 = int64 t
            let tsingleCorrect = single v
            let tdoubleCorrect = double v
            let tint16Correct = int16 v
            let tint32Correct = int32 v
            let tint64Correct = int64 v
            Assert.AreEqual(tsingleCorrect, tsingle)
            Assert.AreEqual(tdoubleCorrect, tdouble)
            Assert.AreEqual(tint16Correct, tint16)
            Assert.AreEqual(tint32Correct, tint32)
            Assert.AreEqual(tint64Correct, tint64)

        for combo in Combos.IntegralAndFloatingPoint do
            let v = 2.
            let t = combo.tensor(v)
            let tsingle = t |> Convert.ToSingle
            let tdouble = t |> Convert.ToDouble
            let tint16 = t |> Convert.ToInt16
            let tint32 = t |> Convert.ToInt32
            let tint64 = t |> Convert.ToInt64
            let tsingleCorrect = single v
            let tdoubleCorrect = double v
            let tint16Correct = int16 v
            let tint32Correct = int32 v
            let tint64Correct = int64 v
            Assert.AreEqual(tsingleCorrect, tsingle)
            Assert.AreEqual(tdoubleCorrect, tdouble)
            Assert.AreEqual(tint16Correct, tint16)
            Assert.AreEqual(tint32Correct, tint32)
            Assert.AreEqual(tint64Correct, tint64)

        for combo in Combos.Bool do
            let v = true
            let t = combo.tensor(v)
            let tbool = t |> Convert.ToBoolean
            let tboolCorrect = v
            Assert.AreEqual(tboolCorrect, tbool)

    [<Test>]
    member _.TestTensorOnehot () =
        for combo in Combos.All do 
            let t0 = combo.onehot(3, 0)
            let t1 = combo.onehot(3, 1)
            let t2 = combo.onehot(3, 2)
            let t0Correct = combo.tensor([1,0,0])
            let t1Correct = combo.tensor([0,1,0])
            let t2Correct = combo.tensor([0,0,1])
            Assert.AreEqual(t0Correct, t0)
            Assert.AreEqual(t1Correct, t1)
            Assert.AreEqual(t2Correct, t2)

    [<Test>]
    // Test the underlying GetItem on the RawPrimal, useful when testing backends
    member _.TestTensorGetItemOnPrimal () =
      for combo in Combos.IntegralAndFloatingPoint do 
        let t0 = combo.tensor(2.)
        Assert.AreEqual(2.0, System.Convert.ToDouble (t0.toScalar()))

        let t1 = combo.tensor([2., 3., 4., 5., 6.])
        Assert.AreEqual(2.0, System.Convert.ToDouble (t1.primalRaw.GetItem(0)))
        Assert.AreEqual(3.0, System.Convert.ToDouble (t1.primalRaw.GetItem(1)))
        Assert.AreEqual(4.0, System.Convert.ToDouble (t1.primalRaw.GetItem(2)))
        Assert.AreEqual(5.0, System.Convert.ToDouble (t1.primalRaw.GetItem(3)))
        Assert.AreEqual(6.0, System.Convert.ToDouble (t1.primalRaw.GetItem(4)))

        let t2 = combo.tensor([[2.]; [3.]])
        Assert.AreEqual(2.0, System.Convert.ToDouble (t2.primalRaw.GetItem(0, 0)))
        Assert.AreEqual(3.0, System.Convert.ToDouble (t2.primalRaw.GetItem(1, 0)))

        let t2b = combo.tensor([[1.;2.]; [3.;4.]])
        Assert.AreEqual(1.0, System.Convert.ToDouble (t2b.primalRaw.GetItem(0, 0)))
        Assert.AreEqual(2.0, System.Convert.ToDouble (t2b.primalRaw.GetItem(0, 1)))
        Assert.AreEqual(3.0, System.Convert.ToDouble (t2b.primalRaw.GetItem(1, 0)))
        Assert.AreEqual(4.0, System.Convert.ToDouble (t2b.primalRaw.GetItem(1, 1)))

        let t3 = combo.tensor([[[2.; 3.]]])
        Assert.AreEqual(2.0, System.Convert.ToDouble (t3.primalRaw.GetItem(0, 0, 0)))
        Assert.AreEqual(3.0, System.Convert.ToDouble (t3.primalRaw.GetItem(0, 0, 1)))

        let t4 = combo.tensor([[[[1.]]]])
        Assert.AreEqual(1.0, System.Convert.ToDouble (t4.primalRaw.GetItem(0, 0, 0, 0)))

    [<Test>]
    // Test the underlying GetItem on the RawPrimal, useful when testing backends
    member _.TestTensorGetSliceOnPrimal () =
      for combo in Combos.IntegralAndFloatingPoint do 
        let t0 = combo.tensor(2.)
        Assert.AreEqual(2.0, System.Convert.ToDouble (t0.toScalar()))

        let t1 = combo.tensor([ 0 .. 10 ])
        let t1slice1 = t1.primalRaw.GetSlice(array2D [ [ 3; 4; 0 ] ])
        let t1slice2 = t1.primalRaw.GetSlice(array2D [ [ 3; 3; 0 ] ])

        Assert.AreEqual(3, t1slice1.GetItem(0))
        Assert.AreEqual(4, t1slice1.GetItem(1))
        Assert.AreEqual(1, t1slice1.Dim)
        Assert.AreEqual(2, t1slice1.Shape.[0])

        Assert.AreEqual(3, t1slice2.GetItem(0))
        Assert.AreEqual(1, t1slice2.Dim)
        Assert.AreEqual(1, t1slice2.Shape.[0])

        // TODO: slicing reducing down to scalar
        //let t1slice3 = t1.primalRaw.GetSlice(array2D [ [ 3; 3; 1 ] ])
        //Assert.AreEqual(3, t1slice3.GetItem(0))
        //Assert.AreEqual(0, t1slice3.Dim)

        let t2 = combo.tensor([ for i in 0 .. 10 -> [ i*10 .. i*10+10 ] ])
        let t2slice1 = t2.primalRaw.GetSlice(array2D [ [ 3; 5; 0 ]; [ 3; 5; 0 ] ])

        Assert.AreEqual(33, t2slice1.GetItem(0, 0))
        Assert.AreEqual(34, t2slice1.GetItem(0, 1))
        Assert.AreEqual(35, t2slice1.GetItem(0, 2))
        Assert.AreEqual(43, t2slice1.GetItem(1, 0))
        Assert.AreEqual(44, t2slice1.GetItem(1, 1))
        Assert.AreEqual(45, t2slice1.GetItem(1, 2))
        Assert.AreEqual(53, t2slice1.GetItem(2, 0))
        Assert.AreEqual(54, t2slice1.GetItem(2, 1))
        Assert.AreEqual(55, t2slice1.GetItem(2, 2))

        let t2slice2 = t2.primalRaw.GetSlice(array2D [ [ 3; 5; 0 ]; [ 3; 3; 1 ] ])
        Assert.AreEqual(33, t2slice2.GetItem(0))
        Assert.AreEqual(43, t2slice2.GetItem(1))
        Assert.AreEqual(53, t2slice2.GetItem(2))

        let t2slice3 = t2.primalRaw.GetSlice(array2D [ [ 3; 3; 1 ]; [ 3; 5; 0 ] ])
        Assert.AreEqual(33, t2slice3.GetItem(0))
        Assert.AreEqual(34, t2slice3.GetItem(1))
        Assert.AreEqual(35, t2slice3.GetItem(2))


    [<Test>]
    // Test cases of indexing where indexing returns a scalar
    member _.TestTensorIndexItemAsScalarTensor () =
      for combo in Combos.IntegralAndFloatingPoint do 
        let t0 = combo.tensor(2.)
        Assert.AreEqual(2.0, System.Convert.ToDouble (t0.toScalar()))

        let t1 = combo.tensor([2., 3., 4., 5., 6.])
        let t1_0 = t1.[0]
        let t1_1 = t1.[1]
        let t1_0_s = t1_0.toScalar()
        let t1_1_s = t1_1.toScalar()
        Assert.AreEqual(2.0, System.Convert.ToDouble t1_0_s)
        Assert.AreEqual(3.0, System.Convert.ToDouble t1_1_s)
        Assert.AreEqual(4.0, System.Convert.ToDouble (t1.[2].toScalar()))
        Assert.AreEqual(5.0, System.Convert.ToDouble (t1.[3].toScalar()))

        let t2 = combo.tensor([[2.]; [3.]])
        Assert.AreEqual(2.0, System.Convert.ToDouble (t2.[0,0].toScalar()))
        Assert.AreEqual(3.0, System.Convert.ToDouble (t2.[1,0].toScalar()))

        let t2b = combo.tensor([[1.;2.]; [3.;4.]])
        Assert.AreEqual(1.0, System.Convert.ToDouble (t2b.[0,0].toScalar()))
        Assert.AreEqual(2.0, System.Convert.ToDouble (t2b.[0,1].toScalar()))
        Assert.AreEqual(3.0, System.Convert.ToDouble (t2b.[1,0].toScalar()))
        Assert.AreEqual(4.0, System.Convert.ToDouble (t2b.[1,1].toScalar()))

        let t3 = combo.tensor([[[2.; 3.]]])
        Assert.AreEqual(2.0, System.Convert.ToDouble (t3.[0,0,0].toScalar()))
        Assert.AreEqual(3.0, System.Convert.ToDouble (t3.[0,0,1].toScalar()))

        let t4 = combo.tensor([[[[1.]]]])
        Assert.AreEqual(1.0, System.Convert.ToDouble (t4.[0,0,0,0].toScalar()))

    member _.TestTensorArange () =
        for combo in Combos.All do
            let t = combo.arange(5.)
            let tCorrect = combo.tensor([0.,1.,2.,3.,4.])
            Assert.AreEqual(tCorrect, t)

            let t2 = combo.arange(5., 1.5, 0.5)
            let t2Correct = combo.tensor([1.5,2.,2.5,3.,3.5,4.,4.5])
            Assert.AreEqual(t2Correct, t2)

            let t3 = combo.arange(5)
            let t3Correct = combo.tensor([0,1,2,3,4], dtype=DType.Int32)
            Assert.AreEqual(t3Correct, t3)

    [<Test>]
    member _.TestTensorMultinomial () =
        for combo in Combos.FloatingPoint do
            let p1 = combo.tensor([0.2,0.3,0.5])
            let m1 = dsharp.multinomial(p1, numSamples=5000)
            let m1dtype = m1.dtype
            let m1dtypeCorrect = DType.Int32
            let m1mean = m1.float().mean()
            let m1stddev = m1.float().stddev()
            let m1meanCorrect = dsharp.tensor(1.3001).float()
            let m1stddevCorrect = dsharp.tensor(0.7810).float()
            Assert.AreEqual(m1dtypeCorrect, m1dtype)
            Assert.True(m1meanCorrect.allclose(m1mean, 0.1))
            Assert.True(m1stddevCorrect.allclose(m1stddev, 0.1))

            let p2 = combo.tensor([[0.2,0.3,0.5],[0.8,0.1,0.1]])
            let m2 = dsharp.multinomial(p2, numSamples=5000)
            let m2dtype = m2.dtype
            let m2dtypeCorrect = DType.Int32
            let m2mean = m2.float().mean(dim=1)
            let m2stddev = m2.float().stddev(dim=1)
            let m2meanCorrect = dsharp.tensor([1.3001, 0.3001]).float()
            let m2stddevCorrect = dsharp.tensor([0.7810, 0.6404]).float()
            Assert.AreEqual(m2dtypeCorrect, m2dtype)
            Assert.True(m2meanCorrect.allclose(m2mean, 0.1))
            Assert.True(m2stddevCorrect.allclose(m2stddev, 0.1))

    [<Test>]
    member _.TestTensorToString () =
        for combo in Combos.IntegralAndFloatingPoint do 
            let t0 = combo.tensor(2.)
            let t1 = combo.tensor([[2.]; [2.]])
            let t2 = combo.tensor([[[2.; 2.]]])
            let t3 = combo.tensor([[1.;2.]; [3.;4.]])
            let t4 = combo.tensor([[[[1.]]]])
            let t0String = t0.ToString()
            let t1String = t1.ToString()
            let t2String = t2.ToString()
            let t3String = t3.ToString()
            let t4String = t4.ToString()
            let suffix = 
                match combo.dtype with 
                | Bool -> failwith "unexpected bool dtype in test"
                | Byte -> ""
                | Int8 -> ""
                | Int16 -> ""
                | Int32 -> ""
                | Int64 -> ""
                | Float32 -> ".000000"
                | Float64 -> ".000000"
                | DType.Other _ -> failwith "unexpected user-defined type"
            let t0StringCorrect = sprintf "Tensor 2%s" suffix
            let t1StringCorrect = sprintf "Tensor [[2%s], \n [2%s]]" suffix suffix
            let t2StringCorrect = sprintf "Tensor [[[2%s, 2%s]]]" suffix suffix
            let t3StringCorrect = sprintf "Tensor [[1%s, 2%s], \n [3%s, 4%s]]" suffix suffix suffix suffix
            let t4StringCorrect = sprintf "Tensor [[[[1%s]]]]" suffix
            Assert.AreEqual(t0StringCorrect, t0String)
            Assert.AreEqual(t1StringCorrect, t1String)
            Assert.AreEqual(t2StringCorrect, t2String)
            Assert.AreEqual(t3StringCorrect, t3String)
            Assert.AreEqual(t4StringCorrect, t4String)

        let t0Bool = dsharp.tensor([ 0.5; 1.0 ], dtype=DType.Bool)
        let t0BoolToString = t0Bool.ToString()
        let t0BoolToStringCorrect = sprintf "Tensor [false, true]" 
        Assert.AreEqual(t0BoolToString, t0BoolToStringCorrect)

        let t1Bool = dsharp.tensor([ false; true ], dtype=DType.Bool)
        let t1BoolToString = t1Bool.ToString()
        let t1BoolToStringCorrect = sprintf "Tensor [false, true]" 
        Assert.AreEqual(t1BoolToString, t1BoolToStringCorrect)

    [<Test>]
    member _.TestTensorEqual () =
        for combo in Combos.IntegralAndFloatingPoint do 
            let t1A = combo.tensor(-1.)
            let t1B = combo.tensor(1.)
            let t1C = combo.tensor(1.)
            let t1At1BEqual = t1A = t1B
            let t1At1BEqualCorrect = false
            let t1Bt1CEqual = t1B = t1C
            let t1Bt1CEqualCorrect = true

            Assert.AreEqual(t1At1BEqualCorrect, t1At1BEqual)
            Assert.AreEqual(t1Bt1CEqualCorrect, t1Bt1CEqual)

            // Systematic testing. The tensors below are listed in expected order of comparison
            let t2S =
                [ combo.tensor( 0. )
                  combo.tensor( 1. )
                  combo.tensor([ 1.] )
                  combo.tensor([ 2.] )
                  combo.tensor([ 1.; 1.] )
                  combo.tensor([ 1.; 2. ] )
                  combo.tensor([ 2.; 1. ] ) 
                  combo.tensor([ [ 1.; 1.] ]) ]

            // Check the F# generic '=' gives expected results
            let equalsResults = [| for a in t2S -> [| for b in t2S -> a = b |] |]
            let equalsCorrect = [| for i in 0..t2S.Length-1 -> [| for j in 0..t2S.Length-1 -> (i=j) |] |]

            Assert.AreEqual(equalsResults, equalsCorrect)

    // Bool
        for combo in Combos.Bool do 
            let t1A = combo.tensor(false)
            let t1B = combo.tensor(true)
            let t1C = combo.tensor(true)
            let t1At1BEqual = t1A = t1B
            let t1At1BEqualCorrect = false
            let t1Bt1CEqual = t1B = t1C
            let t1Bt1CEqualCorrect = true

            Assert.AreEqual(t1At1BEqualCorrect, t1At1BEqual)
            Assert.AreEqual(t1Bt1CEqualCorrect, t1Bt1CEqual)

        for combo in Combos.All do 
            for dtype2 in DTypes.All do 
                 if combo.dtype <> dtype2 then 
                     isInvalidOp (fun () -> combo.tensor(1) = combo.tensor(1, dtype=dtype2))

    [<Test>]
    member _.TestTensorHash () =
        for combo in Combos.IntegralAndFloatingPoint do 

            // Systematic testing. The tensors below are listed in expected order of comparison
            let t2S =
                [ combo.tensor( 0. )
                  combo.tensor( 1. )
                  combo.tensor([ 1.] )
                  combo.tensor([ 2.] )
                  combo.tensor([ 1.; 1.] )
                  combo.tensor([ 1.; 2. ] )
                  combo.tensor([ 2.; 1. ] ) 
                  combo.tensor([ [ 1.; 1.] ]) ]

            // Check the F# generic hashes are the same for identical tensors, and different for this small sample of tensors
            let hashSameResults = [| for a in t2S -> [| for b in t2S -> hash a = hash b |] |]
            let hashSameCorrect = [| for i in 0..t2S.Length-1 -> [| for j in 0..t2S.Length-1 -> (i=j) |] |]

            Assert.AreEqual(hashSameResults, hashSameCorrect)

            // Check reallocating an identical tensor doesn't change the hash
            let t2a = combo.tensor([ 1.] )
            let t2b = combo.tensor([ 1.] )
            Assert.AreEqual(t2a.GetHashCode(), t2b.GetHashCode())

            // Check adding `ForwardDiff` doesn't change the hash or equality
            Assert.AreEqual(t2a.forwardDiff(combo.tensor([1.])).GetHashCode(), t2a.GetHashCode())
            Assert.AreEqual(true, (t2a.forwardDiff(combo.tensor([1.]))) = t2a)

            // Check adding `ReverseDiff` doesn't change the hash or equality
            Assert.AreEqual(t2a.reverseDiff().GetHashCode(), t2a.GetHashCode())
            Assert.AreEqual(true, (t2a.reverseDiff()) = t2a)

    [<Test>]
    member _.TestTensorCompare () =
        for combo in Combos.IntegralAndFloatingPoint do 
            let t1A = combo.tensor(2.)
            let t1B = combo.tensor(3.)
            let t1At1BLess = t1A < t1B
            let t1At1BLessCorrect = true

            Assert.AreEqual(t1At1BLessCorrect, t1At1BLess)

    // Bool
        for combo in Combos.Bool do 
            let t1A = combo.tensor(false)
            let t1B = combo.tensor(true)
            let t1At1BLess = t1A < t1B
            let t1At1BLessCorrect = true

            Assert.AreEqual(t1At1BLessCorrect, t1At1BLess)

    [<Test>]
    member _.TestTensorCast () =
        for combo in Combos.IntegralAndFloatingPoint do 
            for dtype2 in DTypes.IntegralAndFloatingPoint do 
                let t1 = combo.tensor([1.; 2.; 3.; 5.])
                let t2 = combo.tensor([1.; 2.; 3.; 5.], dtype=dtype2)
                let t1Cast = t1.cast(dtype2)
                let t2Cast = t2.cast(combo.dtype)

                Assert.AreEqual(t1Cast.dtype, dtype2)
                Assert.AreEqual(t2Cast.dtype, combo.dtype)
                Assert.AreEqual(t1Cast, t2)
                Assert.AreEqual(t1, t2Cast)

        for combo in Combos.IntegralAndFloatingPoint do 
            let t1Bool = combo.tensor([true; false], dtype=DType.Bool)
            let t2Bool = combo.tensor([1.; 0.])
            let t1BoolCast = t1Bool.cast(combo.dtype)
            let t2BoolCast = t2Bool.cast(DType.Bool)

            Assert.AreEqual(t1BoolCast.dtype, combo.dtype)
            Assert.AreEqual(t2BoolCast.dtype, DType.Bool)
            Assert.AreEqual(t1BoolCast, t2Bool)
            Assert.AreEqual(t1Bool, t2BoolCast)

        for combo in Combos.IntegralAndFloatingPoint do 
            let t1 = combo.tensor([1.; 2.; 3.; 5.])
            let t2 = combo.tensor([1.; 2.; 3.; 5.], dtype=DType.Int8)
            let t1Cast = t1.int8()

            Assert.AreEqual(t1Cast.dtype, DType.Int8)
            Assert.AreEqual(t1Cast, t2)

            let t1 = combo.tensor([1.; 2.; 3.; 5.])
            let t2 = combo.tensor([1.; 2.; 3.; 5.], dtype=DType.Int16)
            let t1Cast = t1.int16()

            Assert.AreEqual(t1Cast.dtype, DType.Int16)
            Assert.AreEqual(t1Cast, t2)

            let t1 = combo.tensor([1.; 2.; 3.; 5.])
            let t2 = combo.tensor([1.; 2.; 3.; 5.], dtype=DType.Int32)
            let t1Cast = t1.int32()

            Assert.AreEqual(t1Cast.dtype, DType.Int32)
            Assert.AreEqual(t1Cast, t2)

            let t1 = combo.tensor([1.; 2.; 3.; 5.])
            let t2 = combo.tensor([1.; 2.; 3.; 5.], dtype=DType.Int32)
            let t1Cast = t1.int()

            Assert.AreEqual(t1Cast.dtype, DType.Int32)
            Assert.AreEqual(t1Cast, t2)

            let t1 = combo.tensor([1.; 2.; 3.; 5.])
            let t2 = combo.tensor([1.; 2.; 3.; 5.], dtype=DType.Int64)
            let t1Cast = t1.int64()

            Assert.AreEqual(t1Cast.dtype, DType.Int64)
            Assert.AreEqual(t1Cast, t2)

            let t1 = combo.tensor([1.; 2.; 3.; 5.])
            let t2 = combo.tensor([1.; 2.; 3.; 5.], dtype=DType.Float32)
            let t1Cast = t1.float32()

            Assert.AreEqual(t1Cast.dtype, DType.Float32)
            Assert.AreEqual(t1Cast, t2)

            let t1 = combo.tensor([1.; 2.; 3.; 5.])
            let t2 = combo.tensor([1.; 2.; 3.; 5.], dtype=DType.Float64)
            let t1Cast = t1.float64()

            Assert.AreEqual(t1Cast.dtype, DType.Float64)
            Assert.AreEqual(t1Cast, t2)

            let t1 = combo.tensor([1.; 2.; 3.; 5.])
            let t2 = combo.tensor([1.; 2.; 3.; 5.], dtype=DType.Float64)
            let t1Cast = t1.float()

            Assert.AreEqual(t1Cast.dtype, DType.Float64)
            Assert.AreEqual(t1Cast, t2)

            let t1 = combo.tensor([1.; 2.; 3.; 5.])
            let t2 = combo.tensor([1.; 2.; 3.; 5.], dtype=DType.Float64)
            let t1Cast = t1.double()

            Assert.AreEqual(t1Cast.dtype, DType.Float64)
            Assert.AreEqual(t1Cast, t2)

            let t1 = combo.tensor([1.; 0.])
            let t2 = combo.tensor([1.; 0.], dtype=DType.Bool)
            let t1Cast = t1.bool()

            Assert.AreEqual(t1Cast.dtype, DType.Bool)
            Assert.AreEqual(t1Cast, t2)

    [<Test>]
    member _.TestTensorBool () =
        for tys in Combos.Bool do
            let t1 = tys.tensor([1; 0; 1; 0], dtype=Bool)

            Assert.AreEqual([| true; false; true; false |], t1.toArray())
            Assert.AreEqual(Bool, t1.dtype)

            let t2 = tys.tensor([true; false; true; false], dtype=Bool)

            Assert.AreEqual([| true; false; true; false |], t2.toArray())
            Assert.AreEqual(Bool, t2.dtype)

    [<Test>]
    member _.TestTensorLtTT () =
        // Test all non-bool types
        for combo in Combos.IntegralAndFloatingPoint do 
            let t1 = combo.tensor([1.; 2.; 3.; 5.])
            let t2 = combo.tensor([1.; 3.; 5.; 4.])
            let t1t2Lt = t1.lt(t2)
            let t1t2LtCorrect = combo.tensor([0.; 1.; 1.; 0.], dtype=DType.Bool)

            Assert.AreEqual(t1t2LtCorrect, t1t2Lt)
            Assert.AreEqual(DType.Bool, t1t2Lt.dtype)

        for combo in Combos.Bool do 
            // Test bool type separately
            let t1Bool = combo.tensor([true; true; false; false ])
            let t2Bool = combo.tensor([true; false; true; false ])
            let t1Boolt2BoolLt = t1Bool.lt(t2Bool)
            let t1Boolt2BoolLtCorrect = combo.tensor([false; false; true; false ], dtype=DType.Bool)

            Assert.AreEqual(t1Boolt2BoolLtCorrect, t1Boolt2BoolLt)

    [<Test>]
    member _.TestTensorLeTT () =
        // Test all non-bool types
        for combo in Combos.IntegralAndFloatingPoint do 
            let t1 = combo.tensor([1.; 2.; 3.; 5.])
            let t2 = combo.tensor([1.; 3.; 5.; 4.])
            let t1t2Le = t1.le(t2)
            let t1t2LeCorrect = combo.tensor([1.; 1.; 1.; 0.], dtype=DType.Bool)

            Assert.AreEqual(t1t2LeCorrect, t1t2Le)
            Assert.AreEqual(DType.Bool, t1t2Le.dtype)

        // Test bool type separately
        for combo in Combos.Bool do 
            let t1Bool = combo.tensor([true; true; false; false ])
            let t2Bool = combo.tensor([true; false; true; false ])
            let t1Boolt2BoolLe = t1Bool.le(t2Bool)
            let t1Boolt2BoolLeCorrect = combo.tensor([true; false; true; true ], dtype=DType.Bool)

            Assert.AreEqual(t1Boolt2BoolLeCorrect, t1Boolt2BoolLe)

    [<Test>]
    member _.TestTensorGtTT () =
        // Test all non-bool types
        for combo in Combos.IntegralAndFloatingPoint do 
            let t1 = combo.tensor([1.; 2.; 3.; 5.])
            let t2 = combo.tensor([1.; 3.; 5.; 4.])
            let t1t2Gt = t1.gt(t2)
            let t1t2GtCorrect = combo.tensor([0.; 0.; 0.; 1.], dtype=DType.Bool)

            Assert.AreEqual(t1t2GtCorrect, t1t2Gt)
            Assert.AreEqual(DType.Bool, t1t2Gt.dtype)

        // Test bool type separately
        for combo in Combos.Bool do 
            let t1Bool = combo.tensor([true; true; false; false ])
            let t2Bool = combo.tensor([true; false; true; false ])
            let t1Boolt2BoolGt = t1Bool.gt(t2Bool)
            let t1Boolt2BoolGtCorrect = combo.tensor([false; true; false; false ], dtype=DType.Bool)

            Assert.AreEqual(t1Boolt2BoolGtCorrect, t1Boolt2BoolGt)

    [<Test>]
    member _.TestTensorGeTT () =
        // Test all non-bool types
        for combo in Combos.IntegralAndFloatingPoint do 
            let t1 = combo.tensor([1.; 2.; 3.; 5.])
            let t2 = combo.tensor([1.; 3.; 5.; 4.])
            let t1t2Ge = t1.ge(t2)
            let t1t2GeCorrect = combo.tensor([1.; 0.; 0.; 1.], dtype=DType.Bool)

            Assert.AreEqual(t1t2GeCorrect, t1t2Ge)
            Assert.AreEqual(DType.Bool, t1t2Ge.dtype)

        // Test bool type separately
        for combo in Combos.Bool do 
            // Test bool type separately
            let t1Bool = combo.tensor([true; true; false; false ])
            let t2Bool = combo.tensor([true; false; true; false ])
            let t1Boolt2BoolGe = t1Bool.ge(t2Bool)
            let t1Boolt2BoolGeCorrect = combo.tensor([true; true; false; true ], dtype=DType.Bool)

            Assert.AreEqual(t1Boolt2BoolGeCorrect, t1Boolt2BoolGe)

    [<Test>]
    member _.TestTensorIsinf () =
        // isinf always returns bool tensor
        for combo in Combos.FloatingPoint do 
            let t = combo.tensor([1.; infinity; 3.; -infinity])
            let i = dsharp.isinf(t)
            let iCorrect = combo.tensor([0.; 1.; 0.; 1.], dtype=DType.Bool)
            Assert.AreEqual(iCorrect, i)

        // Integer tensors always return 0 for isinf
        for combo in Combos.IntegralAndBool do 
            let t = combo.tensor([1.; 0.; 1.])
            let i = dsharp.isinf(t)
            let iCorrect = combo.tensor([0.; 0.; 0.], dtype=DType.Bool)
            Assert.AreEqual(iCorrect, i)

    [<Test>]
    member _.TestTensorIsnan () =
        // isnan always returns bool tensor
        for combo in Combos.FloatingPoint do 
            let t = combo.tensor([1.; nan; 3.; nan])
            let i = dsharp.isnan(t)
            let iCorrect = combo.tensor([false; true; false; true], dtype=DType.Bool)
            Assert.AreEqual(iCorrect, i)

        // Integer and bool tensors always return false for isnan
        for combo in Combos.IntegralAndBool do 
            let t = combo.tensor([1.; 0.; 1.])
            let i = dsharp.isnan(t)
            let iCorrect = combo.tensor([0.; 0.; 0.], dtype=DType.Bool)
            Assert.AreEqual(iCorrect, i)

    [<Test>]
    member _.TestTensorOnesLike () =
        for combo in Combos.All do 
            let t = combo.tensor([1.; 2.; 3.; 4.])
            let i = t.onesLike([2])
            let iCorrect = combo.tensor([1.; 1.])
            Assert.AreEqual(iCorrect, i)

    [<Test>]
    member _.TestTensorZerosLike () =
        for combo in Combos.All do 
            let t = combo.tensor([1.; 2.; 3.; 4.])
            let i = t.zerosLike([2])
            let iCorrect = combo.tensor([0.; 0.])
            Assert.AreEqual(iCorrect, i)

    [<Test>]
    member _.TestTensorFullLike () =
        for combo in Combos.All do 
            let t = combo.tensor([1.; 2.; 3.; 4.])
            let i = t.fullLike([2], 4.0)
            let iCorrect = combo.tensor([4.; 4.])
            Assert.AreEqual(iCorrect, i)

    [<Test>]
    member _.TestTensorZeroLike () =
        for combo in Combos.All do 
            let t = combo.tensor([1.; 2.; 3.; 4.])
            let i = t.zeroLike()
            let iCorrect = combo.tensor(0.)
            Assert.AreEqual(iCorrect, i)

    [<Test>]
    member _.TestTensorOneLike () =
        for combo in Combos.All do 
            let t = combo.tensor([1.; 2.; 3.; 4.])
            let i = t.oneLike()
            let iCorrect = combo.tensor(1.)
            Assert.AreEqual(iCorrect, i)

    [<Test>]
    member _.TestTensorRandLike() =
        for combo in Combos.FloatingPoint do 
            let t = combo.tensor([1.; 2.; 3.; 4.])
            let i = t.randLike([2])
            Assert.AreEqual(i.shape, [|2|])
            Assert.AreEqual(i.dtype, t.dtype)
            Assert.AreEqual(i.dtype, combo.dtype)

        for combo in Combos.Bool do
            let t = combo.tensor([1.; 2.; 3.; 4.])
            isInvalidOp(fun () -> t.randLike([2]))

    [<Test>]
    member _.TestTensorRandnLike() =
        for combo in Combos.FloatingPoint do 
            let t = combo.tensor([1.; 2.; 3.; 4.])
            let i = t.randnLike([2])
            Assert.AreEqual(i.shape, [|2|])
            Assert.AreEqual(i.dtype, t.dtype)
            Assert.AreEqual(i.dtype, combo.dtype)

        for combo in Combos.IntegralAndBool do
            let t = combo.tensor([1.; 2.; 3.; 4.])
            isInvalidOp(fun () -> t.randnLike([2]))

    [<Test>]
    member _.TestTensorHasinf () =
        for combo in Combos.FloatingPoint do 
            let t1 = combo.tensor([1.; infinity; 3.; -infinity])
            let t1i = dsharp.hasinf(t1)
            let t1iCorrect = true
            let t2 = combo.tensor([1.; 2.; 3.; 4.])
            let t2i = dsharp.hasinf(t2)
            let t2iCorrect = false
            Assert.AreEqual(t1iCorrect, t1i)
            Assert.AreEqual(t2iCorrect, t2i)

        for combo in Combos.IntegralAndBool do 
            let t = combo.tensor([1.; 0.; 1.])
            let i = dsharp.hasinf(t)
            let iCorrect = false
            Assert.AreEqual(iCorrect, i)

    [<Test>]
    member _.TestTensorHasnan () =
        for combo in Combos.FloatingPoint do 
            let t1 = combo.tensor([1.; nan; 3.; nan])
            let t1i = dsharp.hasnan(t1)
            let t1iCorrect = true
            let t2 = combo.tensor([1.; 2.; 3.; 4.])
            let t2i = dsharp.hasnan(t2)
            let t2iCorrect = false
            Assert.AreEqual(t1iCorrect, t1i)
            Assert.AreEqual(t2iCorrect, t2i)

        for combo in Combos.IntegralAndBool do 
            let t = combo.tensor([1.; 0.; 1.])
            let i = dsharp.hasnan(t)
            let iCorrect = false
            Assert.AreEqual(iCorrect, i)

    [<Test>]
    member _.TestTensorAddTT () =
        // Test all pairs of non-bool types
        for combo in Combos.IntegralAndFloatingPoint do 
            for dtype2 in DTypes.IntegralAndFloatingPoint do 
                match DType.widen combo.dtype dtype2 with 
                | None -> ()
                | Some dtypeRes -> 
                let t1 = combo.tensor([1.; 2.]) + combo.tensor([3.; 4.], dtype=dtype2)
                let t1Correct = combo.tensor([4.; 6.], dtype=dtypeRes)

                let t2 = combo.tensor([1.; 2.]) + combo.tensor(5., dtype=dtype2)
                let t2Correct = combo.tensor([6.; 7.], dtype=dtypeRes)

                Assert.AreEqual(t1Correct, t1)
                Assert.AreEqual(t2Correct, t2)
                Assert.AreEqual(t1.dtype, dtypeRes)
                Assert.AreEqual(t2.dtype, dtypeRes)

    [<Test>]
    member _.TestTensorAddTTScalarBroadcasting () =
        // Test scalar broadcasting 
        for combo in Combos.IntegralAndFloatingPoint do 
            let t3 = combo.tensor([1.; 2.]) + 5.f
            let t3Correct = combo.tensor([6.; 7.])

            let t4 = combo.tensor([1.; 2.]) + 5.
            let t4Correct = combo.tensor([6.; 7.])

            let t5 = combo.tensor([1.; 2.]) + 5
            let t5Correct = combo.tensor([6.; 7.])

            Assert.AreEqual(t3Correct, t3)
            Assert.AreEqual(t4Correct, t4)
            Assert.AreEqual(t5Correct, t5)
            Assert.AreEqual(t3.dtype, combo.dtype)
            Assert.AreEqual(t4.dtype, combo.dtype)
            Assert.AreEqual(t5.dtype, combo.dtype)

        // Bool tensors support addition returning bool
        //
        //   t = torch.tensor([[True]], dtype=torch.bool)
        //   t + t
        //
        //   tensor([[True]])

        for combo in Combos.Bool do 
            let t5a = combo.tensor([true; false])
            let t5b = combo.tensor([true; true])
            let t5 = t5a + t5b
            let t5Correct = combo.tensor([true; true])
            Assert.AreEqual(t5, t5Correct)

    [<Test>]
    member _.TestTensorAddTT_BroadcastingSystematic () =
      for combo in Combos.IntegralAndFloatingPoint do 

        // Check all broadcasts into 2x2
        // 2x2 * 1  (broadcast --> 2x2)
        // 2x2 * 2  (broadcast --> 2x2)
        // 2x2 * 2x1  (broadcast --> 2x2)
        // 2x2 * 1x2  (broadcast --> 2x2)
        let t6a = combo.tensor([ [1.; 2.]; [3.; 4.] ])
        for t6b in [ combo.tensor([ 5.0 ])
                     combo.tensor([ 5.0; 5.0 ])
                     combo.tensor([ [5.0]; [5.0] ])
                     combo.tensor([ [5.0; 5.0] ]) ] do
            let t6 = t6a + t6b
            let t6Commute = t6b + t6a
            let t6Correct = combo.tensor([ [6.; 7.]; [8.; 9.] ])

            Assert.AreEqual(t6Correct, t6)
            Assert.AreEqual(t6Correct, t6Commute)

        // Systematically do all allowed broadcasts into 2x3x4
        // 2x3x4 + 1  (broadcast --> 2x3x4)
        // 2x3x4 + 4  (broadcast --> 2x3x4)
        // 2x3x4 + 1x1  (broadcast --> 2x3x4)
        // 2x3x4 + 3x1  (broadcast --> 2x3x4)
        // 2x3x4 + 1x4  (broadcast --> 2x3x4)
        // etc.
        let t7a = combo.tensor([ [ [1.; 2.; 3.; 4.]; [5.; 6.; 7.; 8.]; [9.; 10.; 11.; 12.] ];
                                 [ [13.; 14.; 15.; 16.]; [17.; 18.; 19.; 20.]; [21.; 22.; 23.; 24.] ]  ])
        let t7Shapes = 
            [ for i1 in [0;1;2] do
                for i2 in [0;1;3] do
                  for i3 in [0;1;4] do 
                    if i1 <> 2 || i2 <> 3 || i3 <> 4 then
                        [| if i1 <> 0 && i2 <> 0 && i3 <> 0 then yield i1
                           if i2 <> 0 && i3 <> 0 then yield i2
                           if i3 <> 0 then yield i3 |] ]
            |> List.distinct

        let t7Results, t7CommuteResults = 
            [| for shape in t7Shapes do 
                  let t7b = combo.tensor( Util.arrayND shape (fun is -> double (Array.sum is) + 2.0))
                  let t7 = t7a + t7b
                  let t7Commute = t7b + t7a
                  yield (t7b, t7), (t7b, t7Commute) |]
            |> Array.unzip

        let t7Expected =
            [|(combo.tensor 2.,                                                       combo.tensor [[[3., 4., 5., 6.], [7., 8., 9., 10.], [11., 12., 13., 14.]], [[15., 16., 17., 18.], [19., 20., 21., 22.], [23., 24., 25., 26.]]]);
              (combo.tensor [2.],                                                     combo.tensor [[[3., 4., 5., 6.], [7., 8., 9., 10.], [11., 12., 13., 14.]], [[15., 16., 17., 18.], [19., 20., 21., 22.], [23., 24., 25., 26.]]]);
              (combo.tensor [2., 3., 4., 5.],                                         combo.tensor [[[3., 5., 7., 9.], [7., 9., 11., 13.], [11., 13., 15., 17.]], [[15., 17., 19., 21.], [19., 21., 23., 25.], [23., 25., 27., 29.]]]);
              (combo.tensor [[2.]],                                                   combo.tensor [[[3., 4., 5., 6.], [7., 8., 9., 10.], [11., 12., 13., 14.]], [[15., 16., 17., 18.], [19., 20., 21., 22.], [23., 24., 25., 26.]]]);
              (combo.tensor [[2., 3., 4., 5.]],                                       combo.tensor [[[3., 5., 7., 9.], [7., 9., 11., 13.], [11., 13., 15., 17.]], [[15., 17., 19., 21.], [19., 21., 23., 25.], [23., 25., 27., 29.]]]);
              (combo.tensor [[2.], [3.], [4.]],                                       combo.tensor [[[3., 4., 5., 6.], [8., 9., 10., 11.], [13., 14., 15., 16.]], [[15., 16., 17., 18.], [20., 21., 22., 23.], [25., 26., 27., 28.]]]);
              (combo.tensor [[2., 3., 4., 5.], [3., 4., 5., 6.], [4., 5., 6., 7.]],   combo.tensor [[[3., 5., 7., 9.], [8., 10., 12., 14.], [13., 15., 17., 19.]], [[15., 17., 19., 21.], [20., 22., 24., 26.], [25., 27., 29., 31.]]]);
              (combo.tensor [[[2.]]],                                                 combo.tensor [[[3., 4., 5., 6.], [7., 8., 9., 10.], [11., 12., 13., 14.]], [[15., 16., 17., 18.], [19., 20., 21., 22.], [23., 24., 25., 26.]]]);
              (combo.tensor [[[2., 3., 4., 5.]]],                                     combo.tensor [[[3., 5., 7., 9.], [7., 9., 11., 13.], [11., 13., 15., 17.]], [[15., 17., 19., 21.], [19., 21., 23., 25.], [23., 25., 27., 29.]]]);
              (combo.tensor [[[2.], [3.], [4.]]],                                     combo.tensor [[[3., 4., 5., 6.], [8., 9., 10., 11.], [13., 14., 15., 16.]], [[15., 16., 17., 18.], [20., 21., 22., 23.], [25., 26., 27., 28.]]]);
              (combo.tensor [[[2., 3., 4., 5.], [3., 4., 5., 6.], [4., 5., 6., 7.]]], combo.tensor [[[3., 5., 7., 9.], [8., 10., 12., 14.], [13., 15., 17., 19.]], [[15., 17., 19., 21.], [20., 22., 24., 26.], [25., 27., 29., 31.]]]);
              (combo.tensor [[[2.]], [[3.]]],                                         combo.tensor [[[3., 4., 5., 6.], [7., 8., 9., 10.], [11., 12., 13., 14.]], [[16., 17., 18., 19.], [20., 21., 22., 23.], [24., 25., 26., 27.]]]);
              (combo.tensor [[[2., 3., 4., 5.]], [[3., 4., 5., 6.]]],                 combo.tensor [[[3., 5., 7., 9.], [7., 9., 11., 13.], [11., 13., 15., 17.]], [[16., 18., 20., 22.], [20., 22., 24., 26.], [24., 26., 28., 30.]]]);
              (combo.tensor [[[2.], [3.], [4.]], [[3.], [4.], [5.]]],                 combo.tensor [[[3., 4., 5., 6.], [8., 9., 10., 11.], [13., 14., 15., 16.]], [[16., 17., 18., 19.], [21., 22., 23., 24.], [26., 27., 28., 29.]]])|]


        Assert.AreEqual(t7Expected, t7Results)
        Assert.AreEqual(t7Expected, t7CommuteResults)



    [<Test>]
    member _.TestTensorStackTs () =
      for combo in Combos.All do 
        let t0a = combo.tensor(1.)
        let t0b = combo.tensor(3.)
        let t0c = combo.tensor(5.)
        let t0 = Tensor.stack([t0a;t0b;t0c])
        let t0Correct = combo.tensor([1.;3.;5.])

        let t1a = combo.tensor([1.; 2.])
        let t1b = combo.tensor([3.; 4.])
        let t1c = combo.tensor([5.; 6.])
        let t1 = Tensor.stack([t1a;t1b;t1c])

        let t2a = combo.tensor([ [1.; 2.] ])
        let t2b = combo.tensor([ [3.; 4.] ])
        let t2c = combo.tensor([ [5.; 6.] ])
        let t2_dim0 = Tensor.stack([t2a;t2b;t2c], dim=0)
        let t2_dim1 = Tensor.stack([t2a;t2b;t2c], dim=1)
        let t2_dim2 = Tensor.stack([t2a;t2b;t2c], dim=2)
        let t2Correct_dim0 = combo.tensor([[[1.;2.]];[[3.;4.]];[[5.;6.]]])
        let t2Correct_dim1 = combo.tensor([[[1.;2.];[3.;4.];[5.;6.]]])
        let t2Correct_dim2 = combo.tensor([[[1.;3.;5.];[2.;4.;6.]]])

        let t1Correct = combo.tensor([[1.;2.];[3.;4.];[5.;6.]])

        Assert.AreEqual(t0Correct, t0)
        Assert.AreEqual(t1Correct, t1)
        Assert.AreEqual(t0.dtype, combo.dtype)
        Assert.AreEqual(t1.dtype, combo.dtype)

        Assert.AreEqual(t2Correct_dim0, t2_dim0)
        Assert.AreEqual(t2Correct_dim1, t2_dim1)
        Assert.AreEqual(t2Correct_dim2, t2_dim2)

    [<Test>]
    member _.TestTensorUnstackT () =
        for combo in Combos.All do 
            let t0a = combo.tensor(1.)
            let t0b = combo.tensor(3.)
            let t0c = combo.tensor(5.)
            let t0Correct = [t0a;t0b;t0c]
            let t0 = Tensor.stack(t0Correct).unstack()

            let t1a = combo.tensor([1.; 2.])
            let t1b = combo.tensor([3.; 4.])
            let t1c = combo.tensor([5.; 6.])
            let t1Correct = [t1a;t1b;t1c]
            let t1 = Tensor.stack(t1Correct).unstack()

            // 3x1x2
            let t2a = combo.tensor([[[1.;2.]];[[3.;4.]];[[5.;6.]]])
            let t2 = t2a.unstack()
            let t2_dim1 = t2a.unstack(dim=1)
            let t2_dim2 = t2a.unstack(dim=2)
            // 3 of 1x2
            let t2Correct = [combo.tensor [[1.;2.]]; combo.tensor [[3.;4.]]; combo.tensor [[5.;6.]]]
            // 1 of 3x2
            let t2Correct_dim1 = [combo.tensor [[1.;2.];[3.;4.];[5.;6.]]]
            // 2 of 3x1
            let t2Correct_dim2 = [combo.tensor [[1.];[3.];[5.]]; combo.tensor [[2.];[4.];[6.]]]

            Assert.AreEqual(t0Correct, Seq.toList t0)
            Assert.AreEqual(t1Correct, Seq.toList t1)
            for t in t1 do 
                Assert.AreEqual(t.dtype, combo.dtype)
            Assert.AreEqual(t2Correct, t2)
            Assert.AreEqual(t2Correct_dim1, t2_dim1)
            Assert.AreEqual(t2Correct_dim2, t2_dim2)

    [<Test>]
    member _.TestTensorCatTs () =
        for combo in Combos.All do 

            let t0a = combo.tensor([1.; 2.])
            let t0 = Tensor.cat([t0a])
            let t0Correct = combo.tensor([1.;2.])

            Assert.AreEqual(t0Correct, t0)

            let t1a = combo.tensor([1.; 2.]) // 2
            let t1b = combo.tensor([3.; 4.]) // 2
            let t1c = combo.tensor([5.; 6.]) // 2
            let t1 = Tensor.cat([t1a;t1b;t1c]) // 6
            let t1_dim0 = Tensor.cat([t1a;t1b;t1c],dim=0) // 6
            let t1Correct = combo.tensor([1.;2.;3.;4.;5.;6.])

            Assert.AreEqual(t1Correct, t1)
            Assert.AreEqual(t1Correct, t1_dim0)

            let t2a = combo.tensor([ [1.; 2.] ]) // 1x2
            let t2b = combo.tensor([ [3.; 4.] ]) // 1x2
            let t2c = combo.tensor([ [5.; 6.] ]) // 1x2
            let t2 = Tensor.cat([t2a;t2b;t2c]) // 3x2
            let t2_dim0 = Tensor.cat([t2a;t2b;t2c], dim=0) // 3x2
            let t2_dim1 = Tensor.cat([t2a;t2b;t2c], dim=1) // 1x6
            let t2Correct_dim0 = combo.tensor([[1.;2.];[3.;4.];[5.;6.]]) // 3x2
            let t2Correct_dim1 = combo.tensor([[1.;2.;3.;4.;5.;6.]]) // 1x6

            Assert.AreEqual(t2Correct_dim0, t2)
            Assert.AreEqual(t2Correct_dim0, t2_dim0)
            Assert.AreEqual(t2Correct_dim1, t2_dim1)

            // irregular sizes dim0
            let t3a = combo.tensor([ [1.; 2.] ]) // 1x2
            let t3b = combo.tensor([ [3.; 4.];[5.; 6.] ]) // 2x2
            let t3c = combo.tensor([ [7.; 8.] ]) // 1x2
            let t3 = Tensor.cat([t3a;t3b;t3c]) // 4x2
            let t3Correct = combo.tensor([[1.;2.];[3.;4.];[5.;6.];[7.;8.]]) // 4x2

            Assert.AreEqual(t3Correct, t3)

            // irregular sizes dim1
            let t4a = combo.tensor([ [1.]; [2.] ]) // 2x1
            let t4b = combo.tensor([ [3.; 4.];[5.; 6.] ]) // 2x2
            let t4c = combo.tensor([ [7.]; [8.] ]) // 2x1
            let t4_dim1 = Tensor.cat([t4a;t4b;t4c],dim=1) // 2x4
            let t4Correct_dim1 = combo.tensor([[1.;3.;4.;7.];[2.;5.;6.;8.]]) // 2x4

            Assert.AreEqual(t4Correct_dim1, t4_dim1)

    [<Test>]
    member _.TestTensorSplitT_Basics () =
        
        for combo in Combos.All do 
            //6 --> 2;2;2
            let t1in = combo.tensor([1.;2.;3.;4.;5.;6.]) // 6
            let t1 = t1in.split([2;2;2]) |> Seq.toList // 3 of 2
            let t1Correct = [combo.tensor([1.; 2.]);combo.tensor([3.; 4.]);combo.tensor([5.; 6.])]

            Assert.AreEqual(t1Correct, t1)

            // 3x1x2
            let t2in = combo.tensor([[[1.;2.]];[[3.;4.]];[[5.;6.]]])
            let t2 = t2in.split(sizes=[1;1;1], dim=0)  |> Seq.toList // 3 of 1x1x2
            let t2Correct = [combo.tensor [[[1.;2.]]]; combo.tensor [[[3.;4.]]]; combo.tensor [[[5.;6.]]]]

            Assert.AreEqual(t2Correct, t2)

            let t3in = combo.tensor([[[1.;2.]];[[3.;4.]];[[5.;6.]]])
            let t3 = t3in.split(sizes=[1;2], dim=0)  |> Seq.toList // 2 of 1x1x2 and 2x1x2
            let t3Correct = [combo.tensor [[[1.;2.]]]; combo.tensor [[[3.;4.]];[[5.;6.]]]]

            Assert.AreEqual(t3Correct, t3)

            let t4in = combo.tensor([[[1.;2.]];[[3.;4.]];[[5.;6.]]])
            let t4 = t4in.split(sizes=[1], dim=1)  |> Seq.toList // 1 of 3x1x2
            let t4Correct = [combo.tensor [[[1.;2.]];[[3.;4.]];[[5.;6.]]]] // 1 of 3x1x2

            Assert.AreEqual(t4Correct, t4)

            let t5in = combo.tensor([[[1.;2.]];[[3.;4.]];[[5.;6.]]])
            let t5 = t5in.split(sizes=[1;1], dim=2)  |> Seq.toList // 2 of 3x1x1
            let t5Correct = [combo.tensor [[[1.]];[[3.]];[[5.]]]; combo.tensor [[[2.]];[[4.]];[[6.]]]] // 2 of 3x1x1

            Assert.AreEqual(t5Correct, t5)

            //systematic split of 6 
            let t6vs = [1..6]
            let t6in = combo.tensor(t6vs) // 6
            for p1 in 0..6 do
              for p2 in 0..6 do
                for p3 in 0..6 do
                   if p1+p2+p3 = 6 then 
                      let t6 = 
                          t6in.split([if p1 > 0 then p1 
                                      if p2 > 0 then p2
                                      if p3 > 0 then p3])
                          |> Seq.toList 
                      let t6Correct = 
                          [if p1 > 0 then combo.tensor(t6vs.[0..p1-1]);
                           if p2 > 0 then combo.tensor(t6vs.[p1..p1+p2-1]);
                           if p3 > 0 then combo.tensor(t6vs.[p1+p2..])]

                      Assert.AreEqual(t6Correct, t6)


            //systematic split of 2x6 along dim1
            let t7vs1 = [1..6]
            let t7vs2 = [7..12]
            let t7in = combo.tensor([ t7vs1; t7vs2] ) // 2x6
            for p1 in 0..6 do
              for p2 in 0..6 do
                for p3 in 0..6 do
                   if p1+p2+p3 = 6 then 
                      let sizes =
                          [if p1 > 0 then p1 
                           if p2 > 0 then p2
                           if p3 > 0 then p3]
                      let t7 = t7in.split(sizes,dim=1) |> Seq.toList 
                      let t7Correct = 
                          [if p1 > 0 then combo.tensor([ t7vs1.[0..p1-1];     t7vs2.[0..p1-1] ]);
                           if p2 > 0 then combo.tensor([ t7vs1.[p1..p1+p2-1]; t7vs2.[p1..p1+p2-1] ]);
                           if p3 > 0 then combo.tensor([ t7vs1.[p1+p2..];     t7vs2.[p1+p2..] ])]

                      Assert.AreEqual(t7Correct, t7)



    [<Test>]
    member _.TestTensorAddT2T1 () =
        // Test all non-bool types
        for combo in Combos.IntegralAndFloatingPoint do 
            let t1 = combo.tensor([[1.; 2.]; [3.; 4.]]) + combo.tensor([5.; 6.])
            let t1Correct = combo.tensor([[6.; 8.]; [8.; 10.]])

            Assert.AreEqual(t1Correct, t1)
            Assert.AreEqual(t1.dtype, combo.dtype)

        for combo in Combos.Bool do 
            // check broadcast for bool tensor 0 --> [2]
            let t6a = combo.tensor([true; false])
            let t6b = combo.tensor(true)
            let t6 = t6a + t6b
            let t6Correct = combo.tensor([true; true])
            Assert.AreEqual(t6, t6Correct)

            // check broadcast for bool tensor [1] --> [2]
            let t7a = combo.tensor([true; false])
            let t7b = combo.tensor([true])
            let t7 = t7a + t7b
            let t7Correct = combo.tensor([true; true])
            Assert.AreEqual(t7, t7Correct)


    [<Test>]
    member _.TestTensorSubTT () =
        // Test all pairs of non-bool types, for widening
        for combo in Combos.IntegralAndFloatingPoint do 
            for dtype2 in DTypes.IntegralAndFloatingPoint do 
                match DType.widen combo.dtype dtype2 with 
                | None -> ()
                | Some dtypeRes -> 

                let t1 = combo.tensor([1.; 2.]) - combo.tensor([3.; 4.], dtype=dtype2)
                let t1Correct = combo.tensor([-2.; -2.], dtype=dtypeRes)

                Assert.AreEqual(t1Correct, t1)
                Assert.AreEqual(t1.dtype, dtypeRes)

                let t2 = combo.tensor([1.; 2.]) - combo.tensor(5., dtype=dtype2)
                let t2Correct = combo.tensor([-4.; -3.], dtype=dtypeRes)

                Assert.AreEqual(t2Correct, t2)
                Assert.AreEqual(t2.dtype, dtypeRes)

        // Test scalar broadcast
        for combo in Combos.IntegralAndFloatingPoint do 
            let t3 = combo.tensor([1.; 2.]) - 5.f
            let t3Correct = combo.tensor([-4.; -3.])

            Assert.AreEqual(t3Correct, t3)
            Assert.AreEqual(t3.dtype, combo.dtype)

            let t4 = 5. - combo.tensor([1.; 2.])
            let t4Correct = combo.tensor([4.; 3.])

            Assert.AreEqual(t4Correct, t4)
            Assert.AreEqual(t4.dtype, combo.dtype)

            let t5 = combo.tensor([1.; 2.]) - 5
            let t5Correct = combo.tensor([-4.; -3.])

            Assert.AreEqual(t5Correct, t5)
            Assert.AreEqual(t5.dtype, combo.dtype)

        for combo in Combos.Bool do 
            // Bool tensors do not support subtraction
            //
            //   torch.tensor([[True]], dtype=torch.bool) - torch.tensor([[True]], dtype=torch.bool)
            //
            // RuntimeError: Subtraction, the `-` operator, with two bool tensors is not supported. Use the `^` or `logical_xor()` operator instead.

            let t5a = combo.tensor([true; false])
            let t5b = combo.tensor([true; true])
            isInvalidOp(fun () -> t5a - t5b)

    [<Test>]
    member _.TestTensorMulTT () =
        // Test all pairs of non-bool types
        for combo in Combos.IntegralAndFloatingPoint do 
            for dtype2 in DTypes.IntegralAndFloatingPoint do 
                match DType.widen combo.dtype dtype2 with 
                | None -> ()
                | Some dtypeRes -> 
                let t1 = combo.tensor([1.; 2.]) * combo.tensor([3.; 4.], dtype=dtype2)
                let t1Correct = combo.tensor([3.; 8.], dtype=dtypeRes)

                Assert.AreEqual(t1Correct, t1)
                Assert.AreEqual(t1.dtype, dtypeRes)

                let t2 = combo.tensor([1.; 2.]) * combo.tensor(5., dtype=dtype2)
                let t2Correct = combo.tensor([5.; 10.], dtype=dtypeRes)

                Assert.AreEqual(t2Correct, t2)
                Assert.AreEqual(t2.dtype, dtypeRes)

        // Test scalar broadcasting 
        for combo in Combos.IntegralAndFloatingPoint do 
            let t3 = combo.tensor([1.; 2.]) * 5.f
            let t3Correct = combo.tensor([5.; 10.])

            Assert.AreEqual(t3Correct, t3)

            let t4 = 5. * combo.tensor([1.; 2.])
            let t4Correct = combo.tensor([5.; 10.])

            Assert.AreEqual(t4Correct, t4)
            Assert.AreEqual(t3.dtype, combo.dtype)
            Assert.AreEqual(t4.dtype, combo.dtype)

        // Bool tensors support multiplication giving bool tensor
        //
        //    torch.ones(10, dtype=torch.bool) * torch.ones(10, dtype=torch.bool)
        //
        //    tensor([True, True, True, True, True, True, True, True, True, True])
        for combo in Combos.Bool do 
            let t1 = combo.tensor([true; true])
            let t2 = combo.tensor([true; false])
            let i = t1 * t2
            let iCorrect = combo.tensor([true; false])
            Assert.AreEqual(iCorrect, i)

    [<Test>]
    member _.TestTensorMulTT_BroadcastSystematic () =
      for combo in Combos.FloatingPoint do 
        // 2x2 * 1  (broadcast --> 2x2)
        // 2x2 * 2  (broadcast --> 2x2)
        // 2x2 * 2x1  (broadcast --> 2x2)
        // 2x2 * 1x2  (broadcast --> 2x2)
        let t5a = combo.tensor([ [1.; 2.]; [3.; 4.] ])
        for t5b in [ combo.tensor([ 5.0 ])
                     combo.tensor([ 5.0; 5.0 ])
                     combo.tensor([ [5.0]; [5.0] ])
                     combo.tensor([ [5.0; 5.0] ]) ] do
            let t5 = t5a * t5b
            let t5Commute = t5b * t5a
            let t5Correct = combo.tensor([ [5.; 10.]; [15.; 20.] ])

            Assert.AreEqual(t5Correct, t5)
            Assert.AreEqual(t5Correct, t5Commute)

        // Systematically do all allowed broadcasts into 2x3x4
        // 2x3x4 * 1  (broadcast --> 2x3x4)
        // 2x3x4 * 4  (broadcast --> 2x3x4)
        // 2x3x4 * 1x1  (broadcast --> 2x3x4)
        // 2x3x4 * 3x1  (broadcast --> 2x3x4)
        // 2x3x4 * 1x4  (broadcast --> 2x3x4)
        // etc.
        let t6a = combo.tensor([ [ [1.; 2.; 3.; 4.]; [5.; 6.; 7.; 8.]; [9.; 10.; 11.; 12.] ];
                                  [ [13.; 14.; 15.; 16.]; [17.; 18.; 19.; 20.]; [21.; 22.; 23.; 24.] ]  ])

        // These are all the interesting shapes that broadcast into t6a
        let t6Shapes = 
            [ for i1 in [0;1;2] do
                for i2 in [0;1;3] do
                  for i3 in [0;1;4] do 
                    if i1 <> 2 || i2 <> 3 || i3 <> 4 then
                        [| if i1 <> 0 && i2 <> 0 && i3 <> 0 then yield i1
                           if i2 <> 0 && i3 <> 0 then yield i2
                           if i3 <> 0 then yield i3 |] ]
            |> List.distinct

        let t6Results, t6CommuteResults = 
            [| for shape in t6Shapes do 
                  let t6b = combo.tensor( Util.arrayND shape (fun is -> double (Array.sum is) + 2.0))
                  let t6 = t6a * t6b
                  let t6Commute = t6b * t6a
                  yield (t6b, t6 ), (t6b, t6Commute ) |]
            |> Array.unzip

        let t6Expected =
            [|(combo.tensor 2.,                                                      combo.tensor [[[2., 4., 6., 8.], [10., 12., 14., 16.], [18., 20., 22., 24.]], [[26., 28., 30., 32.], [34., 36., 38., 40.], [42., 44., 46., 48.]]]);
              (combo.tensor [2.],                                                    combo.tensor [[[2., 4., 6., 8.], [10., 12., 14., 16.], [18., 20., 22., 24.]], [[26., 28., 30., 32.], [34., 36., 38., 40.], [42., 44., 46., 48.]]]);
              (combo.tensor [2., 3., 4., 5.],                                        combo.tensor [[[2., 6., 12., 20.], [10., 18., 28., 40.], [18., 30., 44., 60.]], [[26., 42., 60., 80.], [34., 54., 76., 100.], [42., 66., 92., 120.]]]);
              (combo.tensor [[2.]],                                                  combo.tensor [[[2., 4., 6., 8.], [10., 12., 14., 16.], [18., 20., 22., 24.]], [[26., 28., 30., 32.], [34., 36., 38., 40.], [42., 44., 46., 48.]]]);
              (combo.tensor [[2., 3., 4., 5.]],                                      combo.tensor [[[2., 6., 12., 20.], [10., 18., 28., 40.], [18., 30., 44., 60.]], [[26., 42., 60., 80.], [34., 54., 76., 100.], [42., 66., 92., 120.]]]);
              (combo.tensor [[2.], [3.], [4.]],                                      combo.tensor [[[2., 4., 6., 8.], [15., 18., 21., 24.], [36., 40., 44., 48.]], [[26., 28., 30., 32.], [51., 54., 57., 60.], [84., 88., 92., 96.]]]);
              (combo.tensor [[2., 3., 4., 5.], [3., 4., 5., 6.], [4., 5., 6., 7.]],  combo.tensor [[[2., 6., 12., 20.], [15., 24., 35., 48.], [36., 50., 66., 84.]], [[26., 42., 60., 80.], [51., 72., 95., 120.], [84., 110., 138., 168.]]]);
              (combo.tensor [[[2.]]],                                                combo.tensor [[[2., 4., 6., 8.], [10., 12., 14., 16.], [18., 20., 22., 24.]], [[26., 28., 30., 32.], [34., 36., 38., 40.], [42., 44., 46., 48.]]]);
              (combo.tensor [[[2., 3., 4., 5.]]],                                    combo.tensor [[[2., 6., 12., 20.], [10., 18., 28., 40.], [18., 30., 44., 60.]], [[26., 42., 60., 80.], [34., 54., 76., 100.], [42., 66., 92., 120.]]]);
              (combo.tensor [[[2.], [3.], [4.]]],                                    combo.tensor [[[2., 4., 6., 8.], [15., 18., 21., 24.], [36., 40., 44., 48.]], [[26., 28., 30., 32.], [51., 54., 57., 60.], [84., 88., 92., 96.]]]);
              (combo.tensor [[[2., 3., 4., 5.], [3., 4., 5., 6.], [4., 5., 6., 7.]]],combo.tensor [[[2., 6., 12., 20.], [15., 24., 35., 48.], [36., 50., 66., 84.]], [[26., 42., 60., 80.], [51., 72., 95., 120.], [84., 110., 138., 168.]]]);
              (combo.tensor [[[2.]], [[3.]]],                                        combo.tensor [[[2., 4., 6., 8.], [10., 12., 14., 16.], [18., 20., 22., 24.]], [[39., 42., 45., 48.], [51., 54., 57., 60.], [63., 66., 69., 72.]]]);
              (combo.tensor [[[2., 3., 4., 5.]], [[3., 4., 5., 6.]]],                combo.tensor [[[2., 6., 12., 20.],  [10., 18., 28., 40.], [18., 30., 44., 60.]], [[39., 56., 75., 96.], [51., 72., 95., 120.], [63., 88., 115., 144.]]]);
              (combo.tensor [[[2.], [3.], [4.]], [[3.], [4.], [5.]]],                combo.tensor [[[2., 4., 6., 8.],  [15., 18., 21., 24.], [36., 40., 44., 48.]], [[39., 42., 45., 48.], [68., 72., 76., 80.], [105., 110., 115., 120.]]]); |]

        Assert.AreEqual(t6Expected, t6Results)
        Assert.AreEqual(t6Expected, t6CommuteResults)


    [<Test>]
    member _.TestTensorDivTT () =
        for combo in Combos.FloatingPoint do 
            let t1 = combo.tensor([1.; 2.]) / combo.tensor([3.; 4.])
            let t1Correct = combo.tensor([0.333333; 0.5])

            let t2 = combo.tensor([1.; 2.]) / combo.tensor(5.)
            let t2Correct = combo.tensor([0.2; 0.4])

            let t3 = combo.tensor([1.; 2.]) / 5.
            let t3Correct = combo.tensor([0.2; 0.4])

            let t4 = 5. / combo.tensor([1.; 2.])
            let t4Correct = combo.tensor([5.; 2.5])

            Assert.True(t1.allclose(t1Correct, 0.01))
            Assert.True(t2.allclose(t2Correct, 0.01))
            Assert.True(t3.allclose(t3Correct, 0.01))
            Assert.True(t4.allclose(t4Correct, 0.01))
            Assert.AreEqual(t1.dtype, combo.dtype)
            Assert.AreEqual(t2.dtype, combo.dtype)
            Assert.AreEqual(t3.dtype, combo.dtype)
            Assert.AreEqual(t4.dtype, combo.dtype)

        // Integer tensors support integer division
        for combo in Combos.Integral do 
            let t1a = combo.tensor([2; 3; 4])
            let t1b = combo.tensor([1; 2; 3])
            let i1 = t1a / t1b
            let i1Correct = combo.tensor([2; 1; 1])
            Assert.AreEqual(i1Correct, i1)

            let t2a = combo.tensor(6)
            let t2b = combo.tensor([1; 2; 3])
            let i2 = t2a / t2b
            let i2Correct = combo.tensor([6; 3; 2])
            Assert.AreEqual(i2Correct, i2)

            let t3a = combo.tensor([6; 12; 18])
            let t3b = combo.tensor(3)
            let i3 = t3a / t3b
            let i3Correct = combo.tensor([2; 4; 6])
            Assert.AreEqual(i3Correct, i3)

        // Bool tensors don't support /
        //
        //    torch.ones(10, dtype=torch.bool) / torch.ones(10, dtype=torch.bool)
        //
        //    RuntimeError: "div_cpu" not implemented for 'Bool'
        for combo in Combos.Bool do 
            let t2 = combo.tensor([true; false])
            isInvalidOp(fun () -> t2 / t2)

    [<Test>]
    member _.TestTensorPowTT () =
        for combo in Combos.FloatingPoint do 
            let t1 = combo.tensor([1.; 2.]) ** combo.tensor([3.; 4.])
            let t1Correct = combo.tensor([1.; 16.])

            Assert.AreEqual(t1Correct, t1)
            Assert.AreEqual(t1.dtype, combo.dtype)
            let t2 = combo.tensor([1.; 2.]) ** combo.tensor(5.)
            let t2Correct = combo.tensor([1.; 32.])

            Assert.AreEqual(t2Correct, t2)
            Assert.AreEqual(t2.dtype, combo.dtype)

            let t3 = combo.tensor(5.) ** combo.tensor([1.; 2.])
            let t3Correct = combo.tensor([5.; 25.])

            Assert.AreEqual(t3Correct, t3)
            Assert.AreEqual(t3.dtype, combo.dtype)

        for combo in Combos.IntegralAndBool do
            let t1 = combo.tensor([1.0])
            isInvalidOp(fun () -> t1 ** t1)

            let t2a = combo.tensor([1.0])
            let t2b = combo.tensor(1.0)
            isInvalidOp(fun () -> t2a ** t2b)

            let t3a = combo.tensor(1.0)
            let t3b = combo.tensor([1.0])
            isInvalidOp(fun () -> t3a ** t3b)

    [<Test>]
    member _.TestTensorMatMulT2T2 () =
        for combo in Combos.FloatingPoint do 
            let t1 = combo.tensor([[8.0766; 3.3030; 2.1732; 8.9448; 1.1028];
                                   [4.1215; 4.9130; 5.2462; 4.2981; 9.3622];
                                   [7.4682; 5.2166; 5.1184; 1.9626; 0.7562]])
            let t2 = combo.tensor([[5.1067; 0.0681];
                                   [7.4633; 3.6027];
                                   [9.0070; 7.3012];
                                   [2.6639; 2.8728];
                                   [7.9229; 2.3695]])

            let t3 = t1.matmul(t2)
            let t3Correct = combo.tensor([[118.0367; 56.6266];
                                          [190.5926; 90.8155];
                                          [134.3925; 64.1030]])

            Assert.True(t3.allclose(t3Correct, 0.01))
            Assert.AreEqual(t3.dtype, combo.dtype)

        for combo in Combos.Integral do 
            let t1 = combo.tensor([[1; 2]])
            let t2 = combo.tensor([[3]; [4]])

            let t3 = t1.matmul(t2)
            let t3Correct = combo.tensor([[11]])

            Assert.True(t3.allclose(t3Correct, 0.0))
            Assert.AreEqual(t3.dtype, combo.dtype)

        // Matmul of Bool tensor not allowed
        //
        //    t = torch.tensor([[True]], dtype=torch.bool)
        //    t.matmul(t)
        //
        // RuntimeError: _th_mm not supported on CPUType for Bool

        for combo in Combos.Bool do 
            let t3a = combo.tensor([[true]])
            isInvalidOp(fun () -> t3a.matmul(t3a))

    [<Test>]
    member _.TestTensorDot () =
        for combo in Combos.FloatingPoint do 
            let t1 = combo.tensor([8.0766, 3.3030, -2.1732, 8.9448, 1.1028])
            let t2 = combo.tensor([5.1067, -0.0681, 7.4633, -3.6027, 9.0070])
            let t3 = dsharp.dot(t1, t2)
            let t3Correct = combo.tensor(2.5081)
            Assert.True(t3.allclose(t3Correct, 0.01))
            Assert.AreEqual(t3.dtype, combo.dtype)

        for combo in Combos.Integral do 
            let t1 = combo.tensor([1; 2])
            let t2 = combo.tensor([3; 4])

            let t3 = dsharp.dot(t1, t2)
            let t3Correct = combo.tensor(11)

            Assert.True(t3.allclose(t3Correct, 0.0))
            Assert.AreEqual(t3.dtype, combo.dtype)

        for combo in Combos.Bool do 
            let t3a = combo.tensor([true])
            isInvalidOp(fun () -> dsharp.dot(t3a, t3a))

    [<Test>]
    member _.TestTensorDiagonal () =
        for combo in Combos.All do
            let t1 = combo.arange(6.).view([2; 3])
            let t1a = dsharp.diagonal(t1)
            let t1b = dsharp.diagonal(t1, offset=1)
            let t1c = dsharp.diagonal(t1, offset=2)
            let t1d = dsharp.diagonal(t1, offset= -1)
            let t1aCorrect = combo.tensor([0.,4.])
            let t1bCorrect = combo.tensor([1.,5.])
            let t1cCorrect = combo.tensor([2.])
            let t1dCorrect = combo.tensor([3.])
            let t2 = combo.arange(9.).view([3;3])
            let t2a = dsharp.diagonal(t2)
            let t2aCorrect = combo.tensor([0.,4.,8.])
            Assert.AreEqual(t1aCorrect, t1a)
            Assert.AreEqual(t1bCorrect, t1b)
            Assert.AreEqual(t1cCorrect, t1c)
            Assert.AreEqual(t1dCorrect, t1d)
            Assert.AreEqual(t2aCorrect, t2a)

    [<Test>]
    member _.TestTensorTrace () =
        for combo in Combos.FloatingPoint do
            let t1 = combo.arange(6.).view([2; 3])
            let t1a = dsharp.trace(t1)
            let t1aCorrect = combo.tensor(4.)
            let t2 = combo.arange(9.).view([3;3])
            let t2a = dsharp.trace(t2)
            let t2aCorrect = combo.tensor(12.)
            Assert.AreEqual(t1aCorrect, t1a)
            Assert.AreEqual(t2aCorrect, t2a)

        for combo in Combos.Integral do
            let t1 = combo.arange(6.).view([2; 3])
            let t1a = dsharp.trace(t1)
            let t1aCorrect = combo.tensor(4., dtype=DType.Int64)
            let t2 = combo.arange(9.).view([3;3])
            let t2a = dsharp.trace(t2)
            let t2aCorrect = combo.tensor(12., dtype=DType.Int64)
            Assert.AreEqual(t1aCorrect, t1a)
            Assert.AreEqual(t2aCorrect, t2a)

        for combo in Combos.Bool do
            let t1a = combo.tensor([[true]]).trace()
            let t1aCorrect = combo.tensor(1., dtype=DType.Int64)
            Assert.AreEqual(t1aCorrect, t1a)

    [<Test>]
    member _.TestTensorMaxPool1D () =
        for combo in Combos.FloatingPoint do
            let t = combo.tensor([[[-2.1704, -1.1558,  2.5995,  1.3858, -1.3157, -0.3179,  0.9593,  -2.1432,  0.7169, -1.7999],
                                     [ 0.4564, -0.2262,  0.3495,  0.4587, -0.3858,  0.2349,  0.2978,  0.6288,  1.1539,  0.2121]],

                                    [[ 0.6654,  0.7151,  0.9980,  0.1321, -2.0009, -1.1897,  1.0608,  -1.8059, -0.2344,  1.6387],
                                     [ 1.1872, -2.2679, -0.0297, -0.2067, -1.5622, -0.3916,  0.6039,  -1.1469,  0.4560,  1.2069]]])

            let tk3, tk3i = dsharp.maxpool1di(t, 3)
            let tk3Correct = combo.tensor([[[ 2.5995,  1.3858,  0.9593],
                                              [ 0.4564,  0.4587,  1.1539]],
                                     
                                             [[ 0.9980,  0.1321,  1.0608],
                                              [ 1.1872, -0.2067,  0.6039]]])
            let tk3iCorrect = combo.tensor([[[2, 3, 6],
                                              [0, 3, 8]],
                                     
                                             [[2, 3, 6],
                                              [0, 3, 6]]], dtype=DType.Int32)
            Assert.AreEqual(tk3Correct, tk3)
            Assert.AreEqual(tk3iCorrect, tk3i)

            let tk3p1, tk3p1i = dsharp.maxpool1di(t, 3, padding=1)
            let tk3p1Correct = combo.tensor([[[-1.1558,  2.5995,  0.9593,  0.7169],
                                                [ 0.4564,  0.4587,  0.6288,  1.1539]],
                                       
                                               [[ 0.7151,  0.9980,  1.0608,  1.6387],
                                                [ 1.1872, -0.0297,  0.6039,  1.2069]]])
            let tk3p1iCorrect = combo.tensor([[[1, 2, 6, 8],
                                                [0, 3, 7, 8]],
                                       
                                               [[1, 2, 6, 9],
                                                [0, 2, 6, 9]]], dtype=DType.Int32)
            Assert.AreEqual(tk3p1iCorrect, tk3p1i)
            Assert.AreEqual(tk3p1Correct, tk3p1)

            let tk3s2, tk3s2i = dsharp.maxpool1di(t, 3, stride=2)
            let tk3s2Correct = combo.tensor([[[ 2.5995,  2.5995,  0.9593,  0.9593],
                                              [ 0.4564,  0.4587,  0.2978,  1.1539]],
                                     
                                             [[ 0.9980,  0.9980,  1.0608,  1.0608],
                                              [ 1.1872, -0.0297,  0.6039,  0.6039]]])
            let tk3s2iCorrect = combo.tensor([[[2, 2, 6, 6],
                                                  [0, 3, 6, 8]],
                                         
                                                 [[2, 2, 6, 6],
                                                  [0, 2, 6, 6]]], dtype=DType.Int32)
            Assert.AreEqual(tk3s2iCorrect, tk3s2i)
            Assert.AreEqual(tk3s2Correct, tk3s2)

            let tk4s3p2, tk4s3p2i = dsharp.maxpool1di(t, 4, stride=3, padding=2)
            let tk4s3p2Correct = combo.tensor([[[-1.1558,  2.5995,  0.9593,  0.7169],
                                                  [ 0.4564,  0.4587,  0.6288,  1.1539]],
                                         
                                                 [[ 0.7151,  0.9980,  1.0608,  1.6387],
                                                  [ 1.1872, -0.0297,  0.6039,  1.2069]]])
            let tk4s3p2iCorrect = combo.tensor([[[1, 2, 6, 8],
                                                  [0, 3, 7, 8]],
                                         
                                                 [[1, 2, 6, 9],
                                                  [0, 2, 6, 9]]], dtype=DType.Int32)
            Assert.AreEqual(tk4s3p2iCorrect, tk4s3p2i)
            Assert.AreEqual(tk4s3p2Correct, tk4s3p2)

        for combo in Combos.IntegralAndBool do 
            let x = combo.zeros([1;4;4])
            isInvalidOp(fun () -> dsharp.maxpool1d(x,3))

    [<Test>]
    member _.TestTensorMaxPool2D () =
        for combo in Combos.FloatingPoint do
            let t = combo.tensor([[[[ 0.7372,  0.7090,  0.9216,  0.3363,  1.0141, -0.7642,  0.3801, -0.9568],
                                      [-0.3520, -1.2336,  1.8489,  0.9929, -0.8138,  0.0978, -1.3206, -1.5434],
                                      [ 0.6883, -0.2346,  0.1735,  0.6695, -1.9122,  1.1338, -0.1248,  0.2164],
                                      [-1.1349,  0.3008, -0.1635, -1.0362, -0.6487, -0.8422, -0.4334,  1.0604],
                                      [-2.1562, -0.1079,  0.5744, -0.7275,  1.0254, -0.0508, -0.0525, -0.0746],
                                      [-0.7494,  0.6819, -1.7327, -0.4838, -0.6120,  1.6331,  0.1797, -0.6068],
                                      [ 0.6400,  0.1389,  0.3033,  0.3195,  0.9934,  1.2455, -1.0953,  0.9922],
                                      [ 0.2375,  0.6003, -1.1614,  1.0146,  0.2100, -1.0145, -0.1933,  1.1415]],

                                     [[-0.0819,  0.2091,  0.4351,  1.7527, -1.1970,  2.1048,  1.0200, -0.5153],
                                      [ 1.0867, -1.8738, -0.2754, -0.5089,  0.8850, -0.4751, -0.7820,  1.4476],
                                      [-0.9072,  0.9977, -0.9106, -0.3171, -1.2444,  0.7102,  0.5656,  1.2660],
                                      [ 0.1986, -0.4967,  0.2384, -0.6551,  1.0156,  0.0520, -0.1964,  1.1367],
                                      [ 0.8948,  2.2070,  0.9938,  0.5311, -1.0674,  0.3894,  0.4192, -0.6235],
                                      [ 2.7646, -0.6509,  0.4669, -1.8774, -0.6341,  0.5113,  1.2398,  2.5090],
                                      [ 1.0722,  0.8162, -2.3271,  1.3826,  1.3832,  0.6205, -0.9138, -0.8237],
                                      [-0.0688, -1.6786,  0.1672, -0.7255, -0.1228, -0.1603, -2.1906, -2.6372]]],


                                    [[[-1.0461,  0.4063,  0.2085, -0.7598, -1.3893, -0.8866,  1.0594, -0.6184],
                                      [ 2.1120, -0.6475, -0.3964,  0.0378,  0.0138, -0.1672,  0.9265, -1.7734],
                                      [-0.2313,  0.6284, -0.0508, -0.1014, -0.5059,  0.8666, -0.7010, -0.5073],
                                      [ 0.1709,  0.2466,  0.1781, -1.6740, -0.0251, -1.4144, -2.1012,  0.3922],
                                      [ 0.9141,  0.6582, -0.0826, -0.7104,  1.7133,  1.2406,  1.1415, -0.6222],
                                      [-2.1525, -0.2996, -1.3787,  0.0336, -1.4643,  0.6534,  0.3996,  0.3145],
                                      [-0.3298,  0.3855, -0.5100,  1.2770,  0.5306, -0.6604, -0.0489,  0.0609],
                                      [-0.1552, -1.1218, -0.8435,  0.2365,  1.4428,  0.4234, -1.1083, -1.3874]],

                                     [[ 0.0511,  0.1216, -1.0103, -1.2529,  1.7200, -0.0225,  0.7446, -0.8076],
                                      [ 0.2543,  1.4250,  0.7869,  0.0526, -2.1598,  1.8228, -0.4628,  1.4234],
                                      [ 0.5492,  0.8668,  0.2120,  0.6599, -1.0934, -1.3726,  0.4788, -0.1171],
                                      [ 0.5121,  1.2607, -0.4565,  0.5448, -2.5025, -0.5503, -1.3373,  0.1711],
                                      [-0.3939, -0.6382, -0.0899, -1.4706,  0.4580,  0.3304,  1.8958,  0.1178],
                                      [ 0.1109,  0.2468,  0.3485, -0.0960, -0.0432, -0.3026, -1.9750,  0.4057],
                                      [-1.1117, -0.3422,  1.2130, -1.1206,  0.9506, -0.7723,  0.3162, -0.5487],
                                      [ 0.6304, -0.9149,  0.6075, -0.5371,  1.5875, -0.2979, -0.5832, -3.0311]]]])

            let tk3, tk3i = dsharp.maxpool2di(t, 3)
            let tk3Correct = combo.tensor([[[[1.8489, 1.1338],
                                              [0.6819, 1.6331]],

                                             [[1.0867, 2.1048],
                                              [2.7646, 1.0156]]],


                                            [[[2.1120, 0.8666],
                                              [0.9141, 1.7133]],

                                             [[1.4250, 1.8228],
                                              [1.2607, 0.5448]]]])
            let tk3iCorrect = combo.tensor([[[[10, 21],
                                                  [41, 45]],

                                                 [[ 8,  5],
                                                  [40, 28]]],


                                                [[[ 8, 21],
                                                  [32, 36]],

                                                 [[ 9, 13],
                                                  [25, 27]]]], dtype=DType.Int32)
            Assert.AreEqual(tk3Correct, tk3)
            Assert.AreEqual(tk3iCorrect, tk3i)

            let tk3p1, tk3p1i = dsharp.maxpool2di(t, 3, padding=1)
            let tk3p1Correct = combo.tensor([[[[0.7372, 1.8489, 0.3801],
                                                  [0.6883, 1.0254, 1.1338],
                                                  [0.6819, 1.0146, 1.6331]],

                                                 [[1.0867, 1.7527, 2.1048],
                                                  [2.2070, 1.0156, 1.2660],
                                                  [2.7646, 1.3832, 2.5090]]],


                                                [[[2.1120, 0.2085, 1.0594],
                                                  [0.9141, 1.7133, 1.2406],
                                                  [0.3855, 1.4428, 0.6534]],

                                                 [[1.4250, 1.7200, 1.8228],
                                                  [1.2607, 0.6599, 1.8958],
                                                  [0.6304, 1.5875, 0.4057]]]])
            let tk3p1iCorrect = combo.tensor([[[[ 0, 10,  6],
                                                  [16, 36, 21],
                                                  [41, 59, 45]],

                                                 [[ 8,  3,  5],
                                                  [33, 28, 23],
                                                  [40, 52, 47]]],


                                                [[[ 8,  2,  6],
                                                  [32, 36, 37],
                                                  [49, 60, 45]],

                                                 [[ 9,  4, 13],
                                                  [25, 19, 38],
                                                  [56, 60, 47]]]], dtype=DType.Int32)
            Assert.AreEqual(tk3p1iCorrect, tk3p1i)
            Assert.AreEqual(tk3p1Correct, tk3p1)

            let tk3s2, tk3s2i = dsharp.maxpool2di(t, 3, stride=2)
            let tk3s2Correct = combo.tensor([[[[1.8489, 1.8489, 1.1338],
                                                  [0.6883, 1.0254, 1.1338],
                                                  [0.6819, 1.0254, 1.6331]],

                                                 [[1.0867, 1.7527, 2.1048],
                                                  [2.2070, 1.0156, 1.0156],
                                                  [2.7646, 1.3832, 1.3832]]],


                                                [[[2.1120, 0.2085, 1.0594],
                                                  [0.9141, 1.7133, 1.7133],
                                                  [0.9141, 1.7133, 1.7133]],

                                                 [[1.4250, 1.7200, 1.8228],
                                                  [1.2607, 0.6599, 1.8958],
                                                  [1.2130, 1.2130, 1.8958]]]])
            let tk3s2iCorrect = combo.tensor([[[[10, 10, 21],
                                                  [16, 36, 21],
                                                  [41, 36, 45]],

                                                 [[ 8,  3,  5],
                                                  [33, 28, 28],
                                                  [40, 52, 52]]],


                                                [[[ 8,  2,  6],
                                                  [32, 36, 36],
                                                  [32, 36, 36]],

                                                 [[ 9,  4, 13],
                                                  [25, 19, 38],
                                                  [50, 50, 38]]]], dtype=DType.Int32)
            Assert.AreEqual(tk3s2iCorrect, tk3s2i)
            Assert.AreEqual(tk3s2Correct, tk3s2)

            let tk4s3p2, tk4s3p2i = dsharp.maxpool2di(t, 4, stride=3, padding=2)
            let tk4s3p2Correct = combo.tensor([[[[0.7372, 1.8489, 1.0141],
                                                  [0.6883, 1.8489, 1.1338],
                                                  [0.6819, 1.0254, 1.6331]],

                                                 [[1.0867, 1.7527, 2.1048],
                                                  [2.2070, 2.2070, 1.4476],
                                                  [2.7646, 2.2070, 2.5090]]],


                                                [[[2.1120, 0.4063, 1.0594],
                                                  [2.1120, 1.7133, 1.7133],
                                                  [0.9141, 1.7133, 1.7133]],

                                                 [[1.4250, 1.7200, 1.8228],
                                                  [1.4250, 1.4250, 1.8958],
                                                  [0.6304, 1.5875, 1.8958]]]])
            let tk4s3p2iCorrect = combo.tensor([[[[ 0, 10,  4],
                                                      [16, 10, 21],
                                                      [41, 36, 45]],

                                                     [[ 8,  3,  5],
                                                      [33, 33, 15],
                                                      [40, 33, 47]]],


                                                    [[[ 8,  1,  6],
                                                      [ 8, 36, 36],
                                                      [32, 36, 36]],

                                                     [[ 9,  4, 13],
                                                      [ 9,  9, 38],
                                                      [56, 60, 38]]]], dtype=DType.Int32)
            Assert.AreEqual(tk4s3p2iCorrect, tk4s3p2i)
            Assert.AreEqual(tk4s3p2Correct, tk4s3p2)

        for combo in Combos.IntegralAndBool do 
            let x = combo.zeros([4;4;4;4])
            isInvalidOp(fun () -> dsharp.maxpool2d(x,3))

    [<Test>]
    member _.TestTensorMaxPool3D () =
        for combo in Combos.FloatingPoint do
            let t = combo.tensor([[[[ 0.4633,  0.9173,  0.4568, -1.7660, -0.1077],
                                       [-2.1112,  1.5542,  0.5720, -1.0952, -1.8144],
                                       [ 0.3505, -0.9843, -2.5655, -0.9835,  1.2303],
                                       [ 0.8156,  1.5415,  1.3066, -1.1820,  0.2060],
                                       [ 0.0684,  1.5936,  0.2956, -0.5176, -1.6960]],

                                      [[-1.7281, -0.7697, -2.2310,  0.3580,  0.6299],
                                       [ 0.8558, -0.6180, -1.6077, -0.6779,  1.2910],
                                       [ 0.1885, -0.7006, -0.1863, -1.6729, -0.5761],
                                       [ 0.1940, -0.0399,  0.9329,  1.0687,  0.0955],
                                       [-1.0189,  0.4046,  1.1762,  0.3842,  0.6831]],

                                      [[ 0.2996,  0.5738,  0.0369,  0.2835, -0.2363],
                                       [ 0.6847, -0.4949, -0.3974,  0.6808, -1.2942],
                                       [ 1.0910, -0.0594, -0.0037, -0.3355, -1.5056],
                                       [-0.0965,  1.1358,  1.2851, -1.7333, -1.1705],
                                       [ 0.0966, -1.2780,  1.2939,  1.3469, -0.2603]],

                                      [[-0.5270,  1.1442,  0.1259, -1.2813,  0.3536],
                                       [ 0.1579,  0.0828,  1.3531, -0.9110, -0.8747],
                                       [ 0.2473, -0.1507, -0.4880,  0.4575,  1.1186],
                                       [ 2.0900,  1.0479, -0.7209, -1.6928,  1.8761],
                                       [ 2.2015, -0.5097,  0.7364, -1.5177,  0.9212]],

                                      [[ 1.0358,  1.6584, -1.9654, -1.3971,  1.5641],
                                       [ 0.4032,  0.7737,  0.9351, -0.5245,  0.0783],
                                       [-1.2932, -0.9885, -1.1850, -0.7403,  0.1739],
                                       [-0.5471,  0.5017, -1.0571,  1.7574, -0.0911],
                                       [ 0.6944, -1.2772,  0.7473, -1.0983,  1.1462]]],


                                     [[[-1.2563,  0.0688,  1.0405, -0.2582,  0.7333],
                                       [ 2.0711, -0.1815,  0.8876, -0.2907,  1.1195],
                                       [-0.3912,  0.3624,  1.0576, -0.4748, -1.4021],
                                       [ 1.2176, -0.6160, -0.3471,  1.1689,  0.5677],
                                       [-0.0639,  0.3765, -0.2614,  1.8267,  0.0315]],

                                      [[ 1.2927,  1.0709, -0.8808,  0.8106, -0.5315],
                                       [ 0.7614, -0.3935,  1.2451, -0.0598, -0.5887],
                                       [-0.4089, -0.8598,  0.2478,  0.1282, -0.2745],
                                       [-0.4139, -1.2905, -0.2625, -2.0453,  1.8941],
                                       [-0.2400, -1.2830, -0.3503, -0.8536, -0.5927]],

                                      [[ 0.8200,  1.8860, -0.5216, -0.9590, -0.9760],
                                       [-1.5796,  2.2379, -0.5714, -1.5612,  1.4035],
                                       [-0.6434, -1.2257,  0.1408,  0.3781, -2.2344],
                                       [ 0.4963,  0.2431,  0.6835,  0.0047,  1.3374],
                                       [-1.5899,  2.5382,  0.9503,  1.9080,  1.8315]],

                                      [[ 0.5853,  1.9343, -0.7472,  2.1774, -2.1895],
                                       [-0.6187, -0.2870,  1.2485,  2.4069, -0.2632],
                                       [-1.6047, -0.3379,  0.5372,  1.7098,  1.6220],
                                       [ 0.5255,  0.2564, -1.8615,  1.5519, -0.5655],
                                       [-0.9452, -1.1828, -1.8192,  1.1349,  0.9806]],

                                      [[-1.8198,  0.5455,  1.1761,  1.3070, -0.4654],
                                       [ 1.2673,  0.2608,  0.8385, -1.0407, -0.6288],
                                       [-0.3860,  1.3343,  1.3084,  0.5794,  0.4639],
                                       [ 0.4750, -0.9006, -1.5002,  0.8689, -0.0379],
                                       [ 0.2891,  0.0195, -0.0503, -0.3235,  1.5407]]]]).unsqueeze(0)

            let tk2, tk2i = dsharp.maxpool3di(t, 2)
            let tk2Correct = combo.tensor([[[[1.5542, 0.5720],
                                                [1.5415, 1.3066]],
                                     
                                               [[1.1442, 1.3531],
                                                [2.0900, 1.2851]]],
                                     
                                     
                                              [[[2.0711, 1.2451],
                                                [1.2176, 1.1689]],
                                     
                                               [[2.2379, 2.4069],
                                                [0.5255, 1.7098]]]]).unsqueeze(0)
            let tk2iCorrect = combo.tensor([[[[ 6,  7],
                                                [16, 17]],
                                     
                                               [[76, 82],
                                                [90, 67]]],
                                     
                                     
                                              [[[ 5, 32],
                                                [15, 18]],
                                     
                                               [[56, 83],
                                                [90, 88]]]], dtype=DType.Int32).unsqueeze(0)
            Assert.AreEqual(tk2Correct, tk2)
            Assert.AreEqual(tk2iCorrect, tk2i)

            let tk2p1, tk2p1i = dsharp.maxpool3di(t, 2, padding=1)
            let tk2p1Correct = combo.tensor([[[[ 0.4633,  0.9173, -0.1077],
                                                [ 0.3505,  1.5542,  1.2303],
                                                [ 0.8156,  1.5936,  0.2060]],
                                     
                                               [[ 0.2996,  0.5738,  0.6299],
                                                [ 1.0910, -0.0037,  1.2910],
                                                [ 0.1940,  1.2939,  1.3469]],
                                     
                                               [[ 1.0358,  1.6584,  1.5641],
                                                [ 0.4032,  1.3531,  1.1186],
                                                [ 2.2015,  1.0479,  1.8761]]],
                                     
                                     
                                              [[[-1.2563,  1.0405,  0.7333],
                                                [ 2.0711,  1.0576,  1.1195],
                                                [ 1.2176,  0.3765,  1.8267]],
                                     
                                               [[ 1.2927,  1.8860,  0.8106],
                                                [ 0.7614,  2.2379,  1.4035],
                                                [ 0.4963,  2.5382,  1.9080]],
                                     
                                               [[ 0.5853,  1.9343,  2.1774],
                                                [ 1.2673,  1.3343,  2.4069],
                                                [ 0.5255,  0.2564,  1.5519]]]]).unsqueeze(0)
            let tk2p1iCorrect = combo.tensor([[[[  0,   1,   4],
                                                    [ 10,   6,  14],
                                                    [ 15,  21,  19]],
                                         
                                                   [[ 50,  51,  29],
                                                    [ 60,  62,  34],
                                                    [ 40,  72,  73]],
                                         
                                                   [[100, 101, 104],
                                                    [105,  82,  89],
                                                    [ 95,  91,  94]]],
                                         
                                         
                                                  [[[  0,   2,   4],
                                                    [  5,  12,   9],
                                                    [ 15,  21,  23]],
                                         
                                                   [[ 25,  51,  28],
                                                    [ 30,  56,  59],
                                                    [ 65,  71,  73]],
                                         
                                                   [[ 75,  76,  78],
                                                    [105, 111,  83],
                                                    [ 90,  91,  93]]]], dtype=DType.Int32).unsqueeze(0)
            Assert.AreEqual(tk2p1iCorrect, tk2p1i)
            Assert.AreEqual(tk2p1Correct, tk2p1)

            let tk2s3, tk2s3i = dsharp.maxpool3di(t, 2, stride=3)
            let tk2s3Correct = combo.tensor([[[[1.5542, 1.2910],
                                                [1.5936, 1.0687]],
                                     
                                               [[1.6584, 1.5641],
                                                [2.2015, 1.8761]]],
                                     
                                     
                                              [[[2.0711, 1.1195],
                                                [1.2176, 1.8941]],
                                     
                                               [[1.9343, 2.4069],
                                                [0.5255, 1.5519]]]]).unsqueeze(0)
            let tk2s3iCorrect = combo.tensor([[[[  6,  34],
                                                    [ 21,  43]],
                                         
                                                   [[101, 104],
                                                    [ 95,  94]]],
                                         
                                         
                                                  [[[  5,   9],
                                                    [ 15,  44]],
                                         
                                                   [[ 76,  83],
                                                    [ 90,  93]]]], dtype=DType.Int32).unsqueeze(0)
            Assert.AreEqual(tk2s3iCorrect, tk2s3i)
            Assert.AreEqual(tk2s3Correct, tk2s3)

            let tk2s3p1, tk2s3p1i = dsharp.maxpool3di(t, 2, stride=3, padding=1)
            let tk2s3p1Correct = combo.tensor([[[[ 0.4633,  0.4568],
                                                    [ 0.8156,  1.3066]],
                                         
                                                   [[ 0.2996,  0.2835],
                                                    [ 2.0900,  1.2851]]],
                                         
                                         
                                                  [[[-1.2563,  1.0405],
                                                    [ 1.2176,  1.1689]],
                                         
                                                   [[ 0.8200,  2.1774],
                                                    [ 0.5255,  1.7098]]]]).unsqueeze(0)
            let tk2s3p1iCorrect = combo.tensor([[[[ 0,  2],
                                                    [15, 17]],
                                         
                                                   [[50, 53],
                                                    [90, 67]]],
                                         
                                         
                                                  [[[ 0,  2],
                                                    [15, 18]],
                                         
                                                   [[50, 78],
                                                    [90, 88]]]], dtype=DType.Int32).unsqueeze(0)
            Assert.AreEqual(tk2s3p1iCorrect, tk2s3p1i)
            Assert.AreEqual(tk2s3p1Correct, tk2s3p1)

        for combo in Combos.IntegralAndBool do 
            let x = combo.zeros([4;4;4;4;4])
            isInvalidOp(fun () -> dsharp.maxpool3d(x,3))

    [<Test>]
    member _.TestTensorMaxUnpool1D () =
        for combo in Combos.FloatingPoint do
            let tk3 = combo.tensor([[[ 2.5995,  1.3858,  0.9593],
                                      [ 0.4564,  0.4587,  1.1539]],
                             
                                     [[ 0.9980,  0.1321,  1.0608],
                                      [ 1.1872, -0.2067,  0.6039]]])
            let tk3i = combo.tensor([[[2, 3, 6],
                                          [0, 3, 8]],
                                 
                                         [[2, 3, 6],
                                          [0, 3, 6]]], dtype=DType.Int32)
            let tk3u = dsharp.maxunpool1d(tk3, tk3i, 3)
            let tk3uCorrect = combo.tensor([[[ 0.0000,  0.0000,  2.5995,  1.3858,  0.0000,  0.0000,  0.9593,  0.0000,  0.0000],
                                             [ 0.4564,  0.0000,  0.0000,  0.4587,  0.0000,  0.0000,  0.0000,  0.0000,  1.1539]],

                                            [[ 0.0000,  0.0000,  0.9980,  0.1321,  0.0000,  0.0000,  1.0608,  0.0000,  0.0000],
                                             [ 1.1872,  0.0000,  0.0000, -0.2067,  0.0000,  0.0000,  0.6039,  0.0000,  0.0000]]])
            Assert.AreEqual(tk3uCorrect, tk3u)

            let tk3p1 = combo.tensor([[[-1.1558,  2.5995,  0.9593,  0.7169],
                                            [ 0.4564,  0.4587,  0.6288,  1.1539]],
                                   
                                           [[ 0.7151,  0.9980,  1.0608,  1.6387],
                                            [ 1.1872, -0.0297,  0.6039,  1.2069]]])
            let tk3p1i = combo.tensor([[[1, 2, 6, 8],
                                                [0, 3, 7, 8]],
                                       
                                               [[1, 2, 6, 9],
                                                [0, 2, 6, 9]]], dtype=DType.Int32)
            let tk3p1u = dsharp.maxunpool1d(tk3p1, tk3p1i, 3, padding=1)
            let tk3p1uCorrect = combo.tensor([[[ 0.0000, -1.1558,  2.5995,  0.0000,  0.0000,  0.0000,  0.9593,
                                                   0.0000,  0.7169,  0.0000],
                                                 [ 0.4564,  0.0000,  0.0000,  0.4587,  0.0000,  0.0000,  0.0000,
                                                   0.6288,  1.1539,  0.0000]],

                                                [[ 0.0000,  0.7151,  0.9980,  0.0000,  0.0000,  0.0000,  1.0608,
                                                   0.0000,  0.0000,  1.6387],
                                                 [ 1.1872,  0.0000, -0.0297,  0.0000,  0.0000,  0.0000,  0.6039,
                                                   0.0000,  0.0000,  1.2069]]])
            Assert.AreEqual(tk3p1uCorrect, tk3p1u)

            let tk3s2 = combo.tensor([[[ 2.5995,  2.5995,  0.9593,  0.9593],
                                              [ 0.4564,  0.4587,  0.2978,  1.1539]],
                                     
                                             [[ 0.9980,  0.9980,  1.0608,  1.0608],
                                              [ 1.1872, -0.0297,  0.6039,  0.6039]]])
            let tk3s2i = combo.tensor([[[2, 2, 6, 6],
                                                  [0, 3, 6, 8]],
                                         
                                                 [[2, 2, 6, 6],
                                                  [0, 2, 6, 6]]], dtype=DType.Int32)
            let tk3s2u = dsharp.maxunpool1d(tk3s2, tk3s2i, 3, stride=2)
            let tk3s2uCorrect = combo.tensor([[[ 0.0000,  0.0000,  2.5995,  0.0000,  0.0000,  0.0000,  0.9593,
                                                   0.0000,  0.0000],
                                                 [ 0.4564,  0.0000,  0.0000,  0.4587,  0.0000,  0.0000,  0.2978,
                                                   0.0000,  1.1539]],

                                                [[ 0.0000,  0.0000,  0.9980,  0.0000,  0.0000,  0.0000,  1.0608,
                                                   0.0000,  0.0000],
                                                 [ 1.1872,  0.0000, -0.0297,  0.0000,  0.0000,  0.0000,  0.6039,
                                                   0.0000,  0.0000]]])
            Assert.AreEqual(tk3s2uCorrect, tk3s2u)

            let tk4s3p2 = combo.tensor([[[-1.1558,  2.5995,  0.9593,  0.7169],
                                              [ 0.4564,  0.4587,  0.6288,  1.1539]],
                                     
                                             [[ 0.7151,  0.9980,  1.0608,  1.6387],
                                              [ 1.1872, -0.0297,  0.6039,  1.2069]]])
            let tk4s3p2i = combo.tensor([[[1, 2, 6, 8],
                                                  [0, 3, 7, 8]],
                                         
                                                 [[1, 2, 6, 9],
                                                  [0, 2, 6, 9]]], dtype=DType.Int32)
            let tk4s3p2u = dsharp.maxunpool1d(tk4s3p2, tk4s3p2i, 4, stride=3, padding=2, outputSize=[2;2;10])
            let tk4s3p2uCorrect = combo.tensor([[[ 0.0000, -1.1558,  2.5995,  0.0000,  0.0000,  0.0000,  0.9593,
                                                   0.0000,  0.7169,  0.0000],
                                                 [ 0.4564,  0.0000,  0.0000,  0.4587,  0.0000,  0.0000,  0.0000,
                                                   0.6288,  1.1539,  0.0000]],

                                                [[ 0.0000,  0.7151,  0.9980,  0.0000,  0.0000,  0.0000,  1.0608,
                                                   0.0000,  0.0000,  1.6387],
                                                 [ 1.1872,  0.0000, -0.0297,  0.0000,  0.0000,  0.0000,  0.6039,
                                                   0.0000,  0.0000,  1.2069]]])
            Assert.AreEqual(tk4s3p2uCorrect, tk4s3p2u)

    [<Test>]
    member _.TestTensorMaxUnpool2D () =
        for combo in Combos.FloatingPoint do
            let tk3 = combo.tensor([[[[1.8489, 1.1338],
                                              [0.6819, 1.6331]],

                                             [[1.0867, 2.1048],
                                              [2.7646, 1.0156]]],


                                            [[[2.1120, 0.8666],
                                              [0.9141, 1.7133]],

                                             [[1.4250, 1.8228],
                                              [1.2607, 0.5448]]]])
            let tk3i = combo.tensor([[[[10, 21],
                                                  [41, 45]],

                                                 [[ 8,  5],
                                                  [40, 28]]],


                                                [[[ 8, 21],
                                                  [32, 36]],

                                                 [[ 9, 13],
                                                  [25, 27]]]], dtype=DType.Int32)
            let tk3u = dsharp.maxunpool2d(tk3, tk3i, 3, outputSize=[2;2;8;8])
            let tk3uCorrect = combo.tensor([[[[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                              [0.0000, 0.0000, 1.8489, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                              [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.1338, 0.0000, 0.0000],
                                              [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                              [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                              [0.0000, 0.6819, 0.0000, 0.0000, 0.0000, 1.6331, 0.0000, 0.0000],
                                              [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                              [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],

                                             [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 2.1048, 0.0000, 0.0000],
                                              [1.0867, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                              [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                              [0.0000, 0.0000, 0.0000, 0.0000, 1.0156, 0.0000, 0.0000, 0.0000],
                                              [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                              [2.7646, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                              [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                              [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]]],


                                            [[[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                              [2.1120, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                              [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.8666, 0.0000, 0.0000],
                                              [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                              [0.9141, 0.0000, 0.0000, 0.0000, 1.7133, 0.0000, 0.0000, 0.0000],
                                              [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                              [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                              [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],

                                             [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                              [0.0000, 1.4250, 0.0000, 0.0000, 0.0000, 1.8228, 0.0000, 0.0000],
                                              [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                              [0.0000, 1.2607, 0.0000, 0.5448, 0.0000, 0.0000, 0.0000, 0.0000],
                                              [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                              [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                              [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                              [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]]]])
            Assert.AreEqual(tk3uCorrect, tk3u)

            let tk3p1 = combo.tensor([[[[0.7372, 1.8489, 0.3801],
                                              [0.6883, 1.0254, 1.1338],
                                              [0.6819, 1.0146, 1.6331]],

                                             [[1.0867, 1.7527, 2.1048],
                                              [2.2070, 1.0156, 1.2660],
                                              [2.7646, 1.3832, 2.5090]]],


                                            [[[2.1120, 0.2085, 1.0594],
                                              [0.9141, 1.7133, 1.2406],
                                              [0.3855, 1.4428, 0.6534]],

                                             [[1.4250, 1.7200, 1.8228],
                                              [1.2607, 0.6599, 1.8958],
                                              [0.6304, 1.5875, 0.4057]]]])
            let tk3p1i = combo.tensor([[[[ 0, 10,  6],
                                                  [16, 36, 21],
                                                  [41, 59, 45]],

                                                 [[ 8,  3,  5],
                                                  [33, 28, 23],
                                                  [40, 52, 47]]],


                                                [[[ 8,  2,  6],
                                                  [32, 36, 37],
                                                  [49, 60, 45]],

                                                 [[ 9,  4, 13],
                                                  [25, 19, 38],
                                                  [56, 60, 47]]]], dtype=DType.Int32)
            let tk3p1u = dsharp.maxunpool2d(tk3p1, tk3p1i, 3, padding=1, outputSize=[2;2;8;8])
            let tk3p1uCorrect = combo.tensor([[[[0.7372, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.3801, 0.0000],
                                                  [0.0000, 0.0000, 1.8489, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [0.6883, 0.0000, 0.0000, 0.0000, 0.0000, 1.1338, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 1.0254, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 0.6819, 0.0000, 0.0000, 0.0000, 1.6331, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 1.0146, 0.0000, 0.0000, 0.0000, 0.0000]],

                                                 [[0.0000, 0.0000, 0.0000, 1.7527, 0.0000, 2.1048, 0.0000, 0.0000],
                                                  [1.0867, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.2660],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 1.0156, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 2.2070, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [2.7646, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 2.5090],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 1.3832, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]]],


                                                [[[0.0000, 0.0000, 0.2085, 0.0000, 0.0000, 0.0000, 1.0594, 0.0000],
                                                  [2.1120, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [0.9141, 0.0000, 0.0000, 0.0000, 1.7133, 1.2406, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.6534, 0.0000, 0.0000],
                                                  [0.0000, 0.3855, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 1.4428, 0.0000, 0.0000, 0.0000]],

                                                 [[0.0000, 0.0000, 0.0000, 0.0000, 1.7200, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 1.4250, 0.0000, 0.0000, 0.0000, 1.8228, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.6599, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 1.2607, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.8958, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.4057],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [0.6304, 0.0000, 0.0000, 0.0000, 1.5875, 0.0000, 0.0000, 0.0000]]]])
            Assert.AreEqual(tk3p1uCorrect, tk3p1u)

            let tk3s2 = combo.tensor([[[[1.8489, 1.8489, 1.1338],
                                              [0.6883, 1.0254, 1.1338],
                                              [0.6819, 1.0254, 1.6331]],

                                             [[1.0867, 1.7527, 2.1048],
                                              [2.2070, 1.0156, 1.0156],
                                              [2.7646, 1.3832, 1.3832]]],


                                            [[[2.1120, 0.2085, 1.0594],
                                              [0.9141, 1.7133, 1.7133],
                                              [0.9141, 1.7133, 1.7133]],

                                             [[1.4250, 1.7200, 1.8228],
                                              [1.2607, 0.6599, 1.8958],
                                              [1.2130, 1.2130, 1.8958]]]])
            let tk3s2i = combo.tensor([[[[10, 10, 21],
                                                  [16, 36, 21],
                                                  [41, 36, 45]],

                                                 [[ 8,  3,  5],
                                                  [33, 28, 28],
                                                  [40, 52, 52]]],


                                                [[[ 8,  2,  6],
                                                  [32, 36, 36],
                                                  [32, 36, 36]],

                                                 [[ 9,  4, 13],
                                                  [25, 19, 38],
                                                  [50, 50, 38]]]], dtype=DType.Int32)
            let tk3s2u = dsharp.maxunpool2d(tk3s2, tk3s2i, 3, stride=2, outputSize=[2;2;8;8])
            let tk3s2uCorrect = combo.tensor([[[[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 1.8489, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [0.6883, 0.0000, 0.0000, 0.0000, 0.0000, 1.1338, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 1.0254, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 0.6819, 0.0000, 0.0000, 0.0000, 1.6331, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],

                                                 [[0.0000, 0.0000, 0.0000, 1.7527, 0.0000, 2.1048, 0.0000, 0.0000],
                                                  [1.0867, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 1.0156, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 2.2070, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [2.7646, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 1.3832, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]]],


                                                [[[0.0000, 0.0000, 0.2085, 0.0000, 0.0000, 0.0000, 1.0594, 0.0000],
                                                  [2.1120, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [0.9141, 0.0000, 0.0000, 0.0000, 1.7133, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],

                                                 [[0.0000, 0.0000, 0.0000, 0.0000, 1.7200, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 1.4250, 0.0000, 0.0000, 0.0000, 1.8228, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.6599, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 1.2607, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.8958, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 1.2130, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]]]])
            Assert.AreEqual(tk3s2uCorrect, tk3s2u)

            let tk4s3p2 = combo.tensor([[[[0.7372, 1.8489, 1.0141],
                                              [0.6883, 1.8489, 1.1338],
                                              [0.6819, 1.0254, 1.6331]],

                                             [[1.0867, 1.7527, 2.1048],
                                              [2.2070, 2.2070, 1.4476],
                                              [2.7646, 2.2070, 2.5090]]],


                                            [[[2.1120, 0.4063, 1.0594],
                                              [2.1120, 1.7133, 1.7133],
                                              [0.9141, 1.7133, 1.7133]],

                                             [[1.4250, 1.7200, 1.8228],
                                              [1.4250, 1.4250, 1.8958],
                                              [0.6304, 1.5875, 1.8958]]]])
            let tk4s3p2i = combo.tensor([[[[ 0, 10,  4],
                                                      [16, 10, 21],
                                                      [41, 36, 45]],

                                                     [[ 8,  3,  5],
                                                      [33, 33, 15],
                                                      [40, 33, 47]]],


                                                    [[[ 8,  1,  6],
                                                      [ 8, 36, 36],
                                                      [32, 36, 36]],

                                                     [[ 9,  4, 13],
                                                      [ 9,  9, 38],
                                                      [56, 60, 38]]]], dtype=DType.Int32)
            let tk4s3p2u = dsharp.maxunpool2d(tk4s3p2, tk4s3p2i, 4, stride=3, padding=2, outputSize=[2;2;8;8])
            let tk4s3p2uCorrect = combo.tensor([[[[0.7372, 0.0000, 0.0000, 0.0000, 1.0141, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 1.8489, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [0.6883, 0.0000, 0.0000, 0.0000, 0.0000, 1.1338, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 1.0254, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 0.6819, 0.0000, 0.0000, 0.0000, 1.6331, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],

                                                 [[0.0000, 0.0000, 0.0000, 1.7527, 0.0000, 2.1048, 0.0000, 0.0000],
                                                  [1.0867, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.4476],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 2.2070, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [2.7646, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 2.5090],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]]],


                                                [[[0.0000, 0.4063, 0.0000, 0.0000, 0.0000, 0.0000, 1.0594, 0.0000],
                                                  [2.1120, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [0.9141, 0.0000, 0.0000, 0.0000, 1.7133, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],

                                                 [[0.0000, 0.0000, 0.0000, 0.0000, 1.7200, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 1.4250, 0.0000, 0.0000, 0.0000, 1.8228, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.8958, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                  [0.6304, 0.0000, 0.0000, 0.0000, 1.5875, 0.0000, 0.0000, 0.0000]]]])
            Assert.AreEqual(tk4s3p2uCorrect, tk4s3p2u)


    [<Test>]
    member _.TestTensorMaxUnpool3D () =
        for combo in Combos.FloatingPoint do
            let tk2 = combo.tensor([[[[1.5542, 0.5720],
                                        [1.5415, 1.3066]],
                             
                                       [[1.1442, 1.3531],
                                        [2.0900, 1.2851]]],
                             
                             
                                      [[[2.0711, 1.2451],
                                        [1.2176, 1.1689]],
                             
                                       [[2.2379, 2.4069],
                                        [0.5255, 1.7098]]]]).unsqueeze(0)
            let tk2i = combo.tensor([[[[ 6,  7],
                                        [16, 17]],
                             
                                       [[76, 82],
                                        [90, 67]]],
                             
                             
                                      [[[ 5, 32],
                                        [15, 18]],
                             
                                       [[56, 83],
                                        [90, 88]]]], dtype=DType.Int32).unsqueeze(0)
            let tk2u = dsharp.maxunpool3d(tk2, tk2i, 2, outputSize=[1;2;5;5;5])
            let tk2uCorrect = combo.tensor([[[[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                               [0.0000, 1.5542, 0.5720, 0.0000, 0.0000],
                                               [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                               [0.0000, 1.5415, 1.3066, 0.0000, 0.0000],
                                               [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],

                                              [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                               [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                               [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                               [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                               [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],

                                              [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                               [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                               [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                               [0.0000, 0.0000, 1.2851, 0.0000, 0.0000],
                                               [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],

                                              [[0.0000, 1.1442, 0.0000, 0.0000, 0.0000],
                                               [0.0000, 0.0000, 1.3531, 0.0000, 0.0000],
                                               [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                               [2.0900, 0.0000, 0.0000, 0.0000, 0.0000],
                                               [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],

                                              [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                               [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                               [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                               [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                               [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]]],


                                             [[[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                               [2.0711, 0.0000, 0.0000, 0.0000, 0.0000],
                                               [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                               [1.2176, 0.0000, 0.0000, 1.1689, 0.0000],
                                               [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],

                                              [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                               [0.0000, 0.0000, 1.2451, 0.0000, 0.0000],
                                               [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                               [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                               [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],

                                              [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                               [0.0000, 2.2379, 0.0000, 0.0000, 0.0000],
                                               [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                               [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                               [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],

                                              [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                               [0.0000, 0.0000, 0.0000, 2.4069, 0.0000],
                                               [0.0000, 0.0000, 0.0000, 1.7098, 0.0000],
                                               [0.5255, 0.0000, 0.0000, 0.0000, 0.0000],
                                               [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],

                                              [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                               [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                               [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                               [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                               [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]]]]).unsqueeze(0)
            Assert.AreEqual(tk2uCorrect, tk2u)

            let tk2p1 = combo.tensor([[[[ 0.4633,  0.9173, -0.1077],
                                                [ 0.3505,  1.5542,  1.2303],
                                                [ 0.8156,  1.5936,  0.2060]],
                                     
                                               [[ 0.2996,  0.5738,  0.6299],
                                                [ 1.0910, -0.0037,  1.2910],
                                                [ 0.1940,  1.2939,  1.3469]],
                                     
                                               [[ 1.0358,  1.6584,  1.5641],
                                                [ 0.4032,  1.3531,  1.1186],
                                                [ 2.2015,  1.0479,  1.8761]]],
                                     
                                     
                                              [[[-1.2563,  1.0405,  0.7333],
                                                [ 2.0711,  1.0576,  1.1195],
                                                [ 1.2176,  0.3765,  1.8267]],
                                     
                                               [[ 1.2927,  1.8860,  0.8106],
                                                [ 0.7614,  2.2379,  1.4035],
                                                [ 0.4963,  2.5382,  1.9080]],
                                     
                                               [[ 0.5853,  1.9343,  2.1774],
                                                [ 1.2673,  1.3343,  2.4069],
                                                [ 0.5255,  0.2564,  1.5519]]]]).unsqueeze(0)
            let tk2p1i = combo.tensor([[[[  0,   1,   4],
                                                    [ 10,   6,  14],
                                                    [ 15,  21,  19]],
                                         
                                                   [[ 50,  51,  29],
                                                    [ 60,  62,  34],
                                                    [ 40,  72,  73]],
                                         
                                                   [[100, 101, 104],
                                                    [105,  82,  89],
                                                    [ 95,  91,  94]]],
                                         
                                         
                                                  [[[  0,   2,   4],
                                                    [  5,  12,   9],
                                                    [ 15,  21,  23]],
                                         
                                                   [[ 25,  51,  28],
                                                    [ 30,  56,  59],
                                                    [ 65,  71,  73]],
                                         
                                                   [[ 75,  76,  78],
                                                    [105, 111,  83],
                                                    [ 90,  91,  93]]]], dtype=DType.Int32).unsqueeze(0)
            let tk2p1u = dsharp.maxunpool3d(tk2p1, tk2p1i, 2, padding=1, outputSize=[1;2;5;5;5])
            let tk2p1uCorrect = combo.tensor([[[[ 0.4633,  0.9173,  0.0000,  0.0000, -0.1077],
                                                   [ 0.0000,  1.5542,  0.0000,  0.0000,  0.0000],
                                                   [ 0.3505,  0.0000,  0.0000,  0.0000,  1.2303],
                                                   [ 0.8156,  0.0000,  0.0000,  0.0000,  0.2060],
                                                   [ 0.0000,  1.5936,  0.0000,  0.0000,  0.0000]],

                                                  [[ 0.0000,  0.0000,  0.0000,  0.0000,  0.6299],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  1.2910],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.1940,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000]],

                                                  [[ 0.2996,  0.5738,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 1.0910,  0.0000, -0.0037,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  1.2939,  1.3469,  0.0000]],

                                                  [[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  1.3531,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  1.1186],
                                                   [ 0.0000,  1.0479,  0.0000,  0.0000,  1.8761],
                                                   [ 2.2015,  0.0000,  0.0000,  0.0000,  0.0000]],

                                                  [[ 1.0358,  1.6584,  0.0000,  0.0000,  1.5641],
                                                   [ 0.4032,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000]]],


                                                 [[[-1.2563,  0.0000,  1.0405,  0.0000,  0.7333],
                                                   [ 2.0711,  0.0000,  0.0000,  0.0000,  1.1195],
                                                   [ 0.0000,  0.0000,  1.0576,  0.0000,  0.0000],
                                                   [ 1.2176,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  0.3765,  0.0000,  1.8267,  0.0000]],

                                                  [[ 1.2927,  0.0000,  0.0000,  0.8106,  0.0000],
                                                   [ 0.7614,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000]],

                                                  [[ 0.0000,  1.8860,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  2.2379,  0.0000,  0.0000,  1.4035],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.4963,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  2.5382,  0.0000,  1.9080,  0.0000]],

                                                  [[ 0.5853,  1.9343,  0.0000,  2.1774,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  2.4069,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.5255,  0.2564,  0.0000,  1.5519,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000]],

                                                  [[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 1.2673,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  1.3343,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000]]]]).unsqueeze(0)
            Assert.AreEqual(tk2p1uCorrect, tk2p1u)

            let tk2s3 = combo.tensor([[[[1.5542, 1.2910],
                                            [1.5936, 1.0687]],
                                 
                                           [[1.6584, 1.5641],
                                            [2.2015, 1.8761]]],
                                 
                                 
                                          [[[2.0711, 1.1195],
                                            [1.2176, 1.8941]],
                                 
                                           [[1.9343, 2.4069],
                                            [0.5255, 1.5519]]]]).unsqueeze(0)
            let tk2s3i = combo.tensor([[[[  6,  34],
                                                    [ 21,  43]],
                                         
                                                   [[101, 104],
                                                    [ 95,  94]]],
                                         
                                         
                                                  [[[  5,   9],
                                                    [ 15,  44]],
                                         
                                                   [[ 76,  83],
                                                    [ 90,  93]]]], dtype=DType.Int32).unsqueeze(0)
            let tk2s3u = dsharp.maxunpool3d(tk2s3, tk2s3i, 2, stride=3, outputSize=[1;2;5;5;5])
            let tk2s3uCorrect = combo.tensor([[[[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                   [0.0000, 1.5542, 0.0000, 0.0000, 0.0000],
                                                   [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                   [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                   [0.0000, 1.5936, 0.0000, 0.0000, 0.0000]],

                                                  [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                   [0.0000, 0.0000, 0.0000, 0.0000, 1.2910],
                                                   [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                   [0.0000, 0.0000, 0.0000, 1.0687, 0.0000],
                                                   [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],

                                                  [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                   [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                   [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                   [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                   [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],

                                                  [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                   [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                   [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                   [0.0000, 0.0000, 0.0000, 0.0000, 1.8761],
                                                   [2.2015, 0.0000, 0.0000, 0.0000, 0.0000]],

                                                  [[0.0000, 1.6584, 0.0000, 0.0000, 1.5641],
                                                   [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                   [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                   [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                   [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]]],


                                                 [[[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                   [2.0711, 0.0000, 0.0000, 0.0000, 1.1195],
                                                   [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                   [1.2176, 0.0000, 0.0000, 0.0000, 0.0000],
                                                   [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],

                                                  [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                   [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                   [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                   [0.0000, 0.0000, 0.0000, 0.0000, 1.8941],
                                                   [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],

                                                  [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                   [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                   [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                   [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                   [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],

                                                  [[0.0000, 1.9343, 0.0000, 0.0000, 0.0000],
                                                   [0.0000, 0.0000, 0.0000, 2.4069, 0.0000],
                                                   [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                   [0.5255, 0.0000, 0.0000, 1.5519, 0.0000],
                                                   [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],

                                                  [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                   [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                   [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                   [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                                   [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]]]]).unsqueeze(0)
            Assert.AreEqual(tk2s3uCorrect, tk2s3u)

            let tk2s3p1 = combo.tensor([[[[ 0.4633,  0.4568],
                                                [ 0.8156,  1.3066]],
                                     
                                               [[ 0.2996,  0.2835],
                                                [ 2.0900,  1.2851]]],
                                     
                                     
                                              [[[-1.2563,  1.0405],
                                                [ 1.2176,  1.1689]],
                                     
                                               [[ 0.8200,  2.1774],
                                                [ 0.5255,  1.7098]]]]).unsqueeze(0)
            let tk2s3p1i = combo.tensor([[[[ 0,  2],
                                                    [15, 17]],
                                         
                                                   [[50, 53],
                                                    [90, 67]]],
                                         
                                         
                                                  [[[ 0,  2],
                                                    [15, 18]],
                                         
                                                   [[50, 78],
                                                    [90, 88]]]], dtype=DType.Int32).unsqueeze(0)
            let tk2s3p1u = dsharp.maxunpool3d(tk2s3p1, tk2s3p1i, 2, stride=3, padding=1, outputSize=[1;2;5;5;5])
            let tk2s3p1uCorrect = combo.tensor([[[[ 0.4633,  0.0000,  0.4568,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.8156,  0.0000,  1.3066,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000]],

                                                  [[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000]],

                                                  [[ 0.2996,  0.0000,  0.0000,  0.2835,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  1.2851,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000]],

                                                  [[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 2.0900,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000]],

                                                  [[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000]]],


                                                 [[[-1.2563,  0.0000,  1.0405,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 1.2176,  0.0000,  0.0000,  1.1689,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000]],

                                                  [[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000]],

                                                  [[ 0.8200,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000]],

                                                  [[ 0.0000,  0.0000,  0.0000,  2.1774,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  1.7098,  0.0000],
                                                   [ 0.5255,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000]],

                                                  [[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000]]]]).unsqueeze(0)
            Assert.AreEqual(tk2s3p1uCorrect, tk2s3p1u)

    [<Test>]
    member _.TestTensorConv1D () =
        for combo in Combos.FloatingPoint do 
            let t1 = combo.tensor([[[0.3460; 0.4414; 0.2384; 0.7905; 0.2267];
                                    [0.5161; 0.9032; 0.6741; 0.6492; 0.8576];
                                    [0.3373; 0.0863; 0.8137; 0.2649; 0.7125];
                                    [0.7144; 0.1020; 0.0437; 0.5316; 0.7366]];

                                    [[0.9871; 0.7569; 0.4329; 0.1443; 0.1515];
                                     [0.5950; 0.7549; 0.8619; 0.0196; 0.8741];
                                     [0.4595; 0.7844; 0.3580; 0.6469; 0.7782];
                                     [0.0130; 0.8869; 0.8532; 0.2119; 0.8120]];

                                    [[0.5163; 0.5590; 0.5155; 0.1905; 0.4255];
                                     [0.0823; 0.7887; 0.8918; 0.9243; 0.1068];
                                     [0.0337; 0.2771; 0.9744; 0.0459; 0.4082];
                                     [0.9154; 0.2569; 0.9235; 0.9234; 0.3148]]])
            let t2 = combo.tensor([[[0.4941; 0.8710; 0.0606];
                                    [0.2831; 0.7930; 0.5602];
                                    [0.0024; 0.1236; 0.4394];
                                    [0.9086; 0.1277; 0.2450]];

                                   [[0.5196; 0.1349; 0.0282];
                                    [0.1749; 0.6234; 0.5502];
                                    [0.7678; 0.0733; 0.3396];
                                    [0.6023; 0.6546; 0.3439]]])

            let t3 = t1.conv1d(t2)
            let t3Correct = combo.tensor([[[2.8516; 2.0732; 2.6420];
                                           [2.3239; 1.7078; 2.7450]];

                                          [[3.0127; 2.9651; 2.5219];
                                           [3.0899; 3.1496; 2.4110]];

                                          [[3.4749; 2.9038; 2.7131];
                                           [2.7692; 2.9444; 3.2554]]])

            Assert.True(t3.allclose(t3Correct, 0.01))

            let t3p1 = t1.conv1d(t2, padding=1)
            let t3p1Correct = combo.tensor([[[1.4392; 2.8516; 2.0732; 2.6420; 2.1177];
                                             [1.4345; 2.3239; 1.7078; 2.7450; 2.1474]];

                                            [[2.4208; 3.0127; 2.9651; 2.5219; 1.2960];
                                             [1.5544; 3.0899; 3.1496; 2.4110; 1.8567]];

                                            [[1.2965; 3.4749; 2.9038; 2.7131; 1.7408];
                                             [1.3549; 2.7692; 2.9444; 3.2554; 1.2120]]])

            Assert.True(t3p1.allclose(t3p1Correct, 0.01))

            let t3p2 = t1.conv1d(t2, padding=2)
            let t3p2Correct = combo.tensor([[[0.6333; 1.4392; 2.8516; 2.0732; 2.6420; 2.1177; 1.0258];
                                             [0.6539; 1.4345; 2.3239; 1.7078; 2.7450; 2.1474; 1.2585]];

                                            [[0.5982; 2.4208; 3.0127; 2.9651; 2.5219; 1.2960; 1.0620];
                                             [0.5157; 1.5544; 3.0899; 3.1496; 2.4110; 1.8567; 1.3182]];

                                            [[0.3165; 1.2965; 3.4749; 2.9038; 2.7131; 1.7408; 0.5275];
                                             [0.3861; 1.3549; 2.7692; 2.9444; 3.2554; 1.2120; 0.7428]]])

            Assert.True(t3p2.allclose(t3p2Correct, 0.01))

            let t3s2 = t1.conv1d(t2, stride=2)
            let t3s2Correct = combo.tensor([[[2.8516; 2.6420];
                                             [2.3239; 2.7450]];

                                            [[3.0127; 2.5219];
                                             [3.0899; 2.4110]];

                                            [[3.4749; 2.7131];
                                             [2.7692; 3.2554]]])

            Assert.True(t3s2.allclose(t3s2Correct, 0.01))

            let t3s3 = t1.conv1d(t2, stride=3)
            let t3s3Correct = combo.tensor([[[2.8516];
                                             [2.3239]];

                                            [[3.0127];
                                             [3.0899]];

                                            [[3.4749];
                                             [2.7692]]])

            Assert.True(t3s3.allclose(t3s3Correct, 0.01))

            let t3s2p1 = t1.conv1d(t2, stride=2, padding=1)
            let t3s2p1Correct = combo.tensor([[[1.4392; 2.0732; 2.1177];
                                                 [1.4345; 1.7078; 2.1474]];

                                                [[2.4208; 2.9651; 1.2960];
                                                 [1.5544; 3.1496; 1.8567]];

                                                [[1.2965; 2.9038; 1.7408];
                                                 [1.3549; 2.9444; 1.2120]]])

            Assert.True(t3s2p1.allclose(t3s2p1Correct, 0.01))

            let t3s3p2 = t1.conv1d(t2, stride=3, padding=2)
            let t3s3p2Correct = combo.tensor([[[0.6333; 2.0732; 1.0258];
                                                 [0.6539; 1.7078; 1.2585]];

                                                [[0.5982; 2.9651; 1.0620];
                                                 [0.5157; 3.1496; 1.3182]];

                                                [[0.3165; 2.9038; 0.5275];
                                                 [0.3861; 2.9444; 0.7428]]])
        
            Assert.True(t3s3p2.allclose(t3s3p2Correct, 0.01))

            let t3d2 = t1.conv1d(t2, dilation=2)
            let t3d2Correct = combo.tensor([[[2.8030];
                                             [2.4735]];

                                            [[2.9226];
                                             [3.1868]];

                                            [[2.8469];
                                             [2.4790]]])

            Assert.True(t3d2.allclose(t3d2Correct, 0.01))

            let t3p2d3 = t1.conv1d(t2, padding=2, dilation=3)
            let t3p2d3Correct = combo.tensor([[[2.1121; 0.8484; 2.2709];
                                                 [1.6692; 0.5406; 1.8381]];

                                                [[2.5078; 1.2137; 0.9173];
                                                 [2.2395; 1.1805; 1.1954]];

                                                [[1.5215; 1.3946; 2.1327];
                                                 [1.0732; 1.3014; 2.0696]]])

            Assert.True(t3p2d3.allclose(t3p2d3Correct, 0.01))

            let t3s3p6d3 = t1.conv1d(t2, stride=3, padding=6, dilation=3)
            let t3s3p6d3Correct = combo.tensor([[[0.6333; 1.5018; 2.2709; 1.0580];
                                                 [0.6539; 1.5130; 1.8381; 1.0479]];

                                                [[0.5982; 1.7459; 0.9173; 0.2709];
                                                 [0.5157; 0.8537; 1.1954; 0.7027]];

                                                [[0.3165; 1.4118; 2.1327; 1.1949];
                                                 [0.3861; 1.5697; 2.0696; 0.8520]]])

            Assert.True(t3s3p6d3.allclose(t3s3p6d3Correct, 0.01))

            let t3b1 = t1.[0].unsqueeze(0).conv1d(t2)
            let t3b1Correct = t3Correct.[0].unsqueeze(0)
            Assert.True(t3b1.allclose(t3b1Correct, 0.01))

            let t3b1s2 = t1.[0].unsqueeze(0).conv1d(t2, stride = 2)
            let t3b1s2Correct = t3s2Correct.[0].unsqueeze(0)

            Assert.True(t3b1s2.allclose(t3b1s2Correct, 0.01))

        for combo in Combos.Integral do 
            let x = combo.ones([1;4;4])
            let y = combo.ones([1;4;4])
            let z = dsharp.conv1d(x,y)
            let zCorrect = combo.tensor([[[16]]])
            Assert.AreEqual(z, zCorrect)

        // check types must always match
        for dtype1 in DTypes.All do 
            for dtype2 in DTypes.All do 
                if dtype1 <> dtype2 then 
                    let x = dsharp.zeros([1;4;4], dtype=dtype1)
                    let y = dsharp.zeros([1;4;4], dtype=dtype2)
                    isException(fun () -> dsharp.conv1d(x,y))

        for combo in Combos.Bool do 
            let x = combo.zeros([1;4;4])
            let y = combo.zeros([1;4;4])
            isInvalidOp(fun () -> dsharp.conv1d(x,y))

    [<Test>]
    member _.TestTensorConv2D () =
        for combo in Combos.FloatingPoint do 
            let t1 = combo.tensor([[[[ 10.7072,  -5.0993,   3.6884,   2.0982],
                                     [ -6.4356,   0.6351,  -2.3156,  -1.3384],
                                     [ -5.1846,   0.6805, -14.1961,   0.8657],
                                     [ -8.8655,  -7.1694,  -3.4903,  -2.9479]],

                                    [[  2.5630,  -2.2935,  -0.8665,   6.7999],
                                     [  1.8098,   3.2082,   2.3160,  -4.7734],
                                     [ 14.7205,   0.9631,   8.1039,   6.7437],
                                     [  3.7847,  -5.9792,  -2.7371,  -7.8548]]],


                                   [[[  3.5499,   0.9546,  -7.5715,   2.8211],
                                     [ -1.2659,   5.2366,  -7.2322,  -5.8877],
                                     [ -2.8041,   2.1746,   2.2397,   0.1242],
                                     [  1.8172,  -0.3989,  -0.2394,   7.1078]],

                                    [[ -3.7765,   2.1584,   6.8627,  -4.1471],
                                     [  4.6748,   7.9756,  -6.0065,   2.0826],
                                     [  5.1038,  -5.5801,  -4.4420,  -2.9498],
                                     [  0.1037,   4.6578,   3.0760,  -4.9566]]]])
            let t2 = combo.tensor([[[[-5.6745, -1.9422,  4.1369],
                                     [ 4.4623,  4.8385,  0.8345],
                                     [ 1.3015,  0.0708,  3.8132]],

                                     [[ 0.9448, -1.9028, -8.0195],
                                      [-5.3200,  0.4264, -1.2142],
                                      [ 1.4442, -7.3623, 14.5340]]],


                                    [[[-3.3486, -3.2725, -3.4595],
                                      [-5.0818, -0.5769, -3.5363],
                                      [ 3.1498,  0.6293, -1.2527]],

                                     [[ 3.2029,  3.9409, 12.6924],
                                      [ 4.1056, -3.2890,  2.4071],
                                      [ 4.2373, -1.8852,  4.4640]]],


                                    [[[ 4.0582, -4.6075,  6.2574],
                                      [-0.9867,  3.4303, -1.9686],
                                      [-5.0618,  5.0045, -2.0878]],

                                     [[ 1.0605, -3.2697, -1.9856],
                                      [-6.5763, -6.3535,  7.2228],
                                      [15.1009,  4.9045,  5.1197]]]])

            let t3 = t1.conv2d(t2)
            let t3Correct = combo.tensor([[[[  10.6089;   -1.4459];
                                            [-132.3437; -165.9882]];

                                             [[  97.8425;   81.2322];
                                              [ 215.2763; -112.2244]];

                                             [[ 427.2891; -101.3674];
                                              [ -35.6012; -168.9572]]];


                                            [[[-127.6157;  -35.6266];
                                              [  -7.7668;  -47.1349]];

                                             [[ 104.2333;   28.7020];
                                              [  27.1404;    8.1246]];

                                             [[-106.0468;  -94.3428];
                                              [ -78.6259;  136.6283]]]])

            let t3p1 = t1.conv2d(t2, padding=1)
            let t3p1Correct = combo.tensor([[[[  86.6988;    8.1164;  -85.8172;   69.5001];
                                              [-154.2592;   10.6089;   -1.4459; -126.2889];
                                              [-176.1860; -132.3437; -165.9882;  -23.2585];
                                              [ -62.8550; -180.0650;  -52.4599;   55.0733]];

                                             [[   3.9697;  -53.5450;   16.3075;  -35.2008];
                                              [ -60.7372;   97.8425;   81.2322;   20.0075];
                                              [  -9.2216;  215.2763; -112.2244;   73.8351];
                                              [  88.4748;  308.1942;  176.2158;  131.2712]];

                                             [[   5.6857;   51.6497;  106.6138;  -17.3603];
                                              [ -46.9604;  427.2891; -101.3674;  226.5788];
                                              [-125.8047;  -35.6012; -168.9572; -141.2721];
                                              [-105.4274; -132.2796;   35.6026;  -13.8173]]];


                                            [[[ 115.1200; -141.3008;   36.3188;  -92.2498];
                                              [-133.0979; -127.6157;  -35.6266;   42.1693];
                                              [  14.0058;   -7.7668;  -47.1349;  116.9311];
                                              [  52.3284;   75.6948;   -3.7964;    3.3106]];

                                             [[  31.6266;  -11.5726;   39.5819;   22.8020];
                                              [ -55.3912;  104.2333;   28.7020;   24.2710];
                                              [  91.6285;   27.1404;    8.1246;   38.5616];
                                              [ -37.8251;  -83.1444; -113.7539;   -7.7113]];

                                             [[  96.3737;  202.0389;  -68.9841;  -74.9820];
                                              [ -11.1773; -106.0468;  -94.3428; -101.9384];
                                              [ -44.8701;  -78.6259;  136.6283;   89.6921];
                                              [  60.9218;   14.3467;  -86.6495;   49.3313]]]])

            let t3p12 = t1.conv2d(t2, paddings=[|1; 2|])
            let t3p12Correct = combo.tensor([[[[   7.5867;   86.6988;    8.1164;  -85.8172;   69.5001;  -35.4485];
                                              [ 210.3501; -154.2592;   10.6089;   -1.4459; -126.2889;   24.8066];
                                              [ -42.1367; -176.1860; -132.3437; -165.9882;  -23.2585;  -44.1093];
                                              [-151.4929;  -62.8550; -180.0650;  -52.4599;   55.0733;   30.0922]];

                                             [[ -15.5535;    3.9697;  -53.5450;   16.3075;  -35.2008;   -7.1871];
                                              [  94.8112;  -60.7372;   97.8425;   81.2322;   20.0075;   33.2591];
                                              [ 127.0036;   -9.2216;  215.2763; -112.2244;   73.8351;  -30.0885];
                                              [ 245.2360;   88.4748;  308.1942;  176.2158;  131.2712;    1.4327]];

                                             [[  20.1355;    5.6857;   51.6497;  106.6138;  -17.3603; -112.0973];
                                              [ 173.8400;  -46.9604;  427.2891; -101.3674;  226.5788;  145.8927];
                                              [ 110.5519; -125.8047;  -35.6012; -168.9572; -141.2721; -159.3897];
                                              [ -16.8828; -105.4274; -132.2796;   35.6026;  -13.8173;   65.2295]]];


                                            [[[  70.6642;  115.1200; -141.3008;   36.3188;  -92.2498;   29.9960];
                                              [ 101.7243; -133.0979; -127.6157;  -35.6266;   42.1693;  -61.3766];
                                              [ -42.8275;   14.0058;   -7.7668;  -47.1349;  116.9311;   53.7170];
                                              [ -51.1392;   52.3284;   75.6948;   -3.7964;    3.3106;   54.5939]];

                                             [[   0.8100;   31.6266;  -11.5726;   39.5819;   22.8020;  -41.0836];
                                              [ -18.1888;  -55.3912;  104.2333;   28.7020;   24.2710;    3.6328];
                                              [  84.1016;   91.6285;   27.1404;    8.1246;   38.5616;   15.0304];
                                              [  68.3032;  -37.8251;  -83.1444; -113.7539;   -7.7113;  -66.3344]];

                                             [[  -7.6892;   96.3737;  202.0389;  -68.9841;  -74.9820;   85.7395];
                                              [  97.9534;  -11.1773; -106.0468;  -94.3428; -101.9384;  -46.0084];
                                              [  21.9169;  -44.8701;  -78.6259;  136.6283;   89.6921; -113.2355];
                                              [ -30.5091;   60.9218;   14.3467;  -86.6495;   49.3313;   22.9582]]]])

            let t3s2 = t1.conv2d(t2, stride=2)
            let t3s2Correct = combo.tensor([[[[  10.6089]];

                                             [[  97.8425]];

                                             [[ 427.2891]]];


                                            [[[-127.6157]];

                                             [[ 104.2333]];

                                             [[-106.0468]]]])

            let t3s13 = t1.conv2d(t2, strides=[|1; 3|])
            let t3s13Correct = combo.tensor([[[[  10.6089];
                                              [-132.3437]];

                                             [[  97.8425];
                                              [ 215.2763]];

                                             [[ 427.2891];
                                              [ -35.6012]]];


                                            [[[-127.6157];
                                              [  -7.7668]];

                                             [[ 104.2333];
                                              [  27.1404]];

                                             [[-106.0468];
                                              [ -78.6259]]]])

            let t3s2p1 = t1.conv2d(t2, stride=2, padding=1)
            let t3s2p1Correct = combo.tensor([[[[  86.6988;  -85.8172];
                                                  [-176.1860; -165.9882]];

                                                 [[   3.9697;   16.3075];
                                                  [  -9.2216; -112.2244]];

                                                 [[   5.6857;  106.6138];
                                                  [-125.8047; -168.9572]]];


                                                [[[ 115.1200;   36.3188];
                                                  [  14.0058;  -47.1349]];

                                                 [[  31.6266;   39.5819];
                                                  [  91.6285;    8.1246]];

                                                 [[  96.3737;  -68.9841];
                                                  [ -44.8701;  136.6283]]]])

            let t3s23p32 = t1.conv2d(t2, strides=[2; 3], paddings=[3; 2])
            let t3s23p32Correct = combo.tensor([[[[   0.0000,    0.0000],
                                                      [   7.5866,  -85.8172],
                                                      [ -42.1364, -165.9885],
                                                      [ -67.0271,   97.8170]],

                                                     [[   0.0000,    0.0000],
                                                      [ -15.5537,   16.3071],
                                                      [ 127.0034, -112.2239],
                                                      [  78.7071,  -84.0060]],

                                                     [[   0.0000,    0.0000],
                                                      [  20.1357,  106.6139],
                                                      [ 110.5519, -168.9587],
                                                      [ -62.9899,  -13.2544]]],


                                                    [[[   0.0000,    0.0000],
                                                      [  70.6642,   36.3191],
                                                      [ -42.8270,  -47.1361],
                                                      [   6.6860,   70.4299]],

                                                     [[   0.0000,    0.0000],
                                                      [   0.8102,   39.5820],
                                                      [  84.1018,    8.1256],
                                                      [  -4.9704,  -58.3407]],

                                                     [[   0.0000,    0.0000],
                                                      [  -7.6887,  -68.9838],
                                                      [  21.9173,  136.6280],
                                                      [  11.1650,   48.6844]]]])
        
            let t3p1d2 = t1.conv2d(t2, padding=1, dilation=2)
            let t3p1d2Correct = combo.tensor([[[[ -72.7697,  -34.7305],
                                                  [ -35.3463, -230.5320]],

                                                 [[ -42.2859,   24.9292],
                                                  [  96.3085,   25.1894]],

                                                 [[-149.3111,   42.9268],
                                                  [  73.8409, -159.8669]]],


                                                [[[ -57.9600,  -88.2215],
                                                  [  50.7950,  -52.7872]],

                                                 [[ -43.4812,   49.7672],
                                                  [ -47.4554,   76.3617]],

                                                 [[ -25.4452,   -9.8843],
                                                  [  35.7940,   27.9557]]]])

            let t3p22d23 = t1.conv2d(t2, paddings=[2;2], dilations=[2;3])
            let t3p22d23Correct = combo.tensor([[[[-3.2693e+01, -4.3192e+01],
                                                      [ 4.7954e+01,  9.6877e+00],
                                                      [ 1.7971e+01, -7.0747e+01],
                                                      [-4.4577e+01, -1.7964e+01]],

                                                     [[ 9.0977e+00, -2.3489e+01],
                                                      [-4.1579e+00, -3.3179e+00],
                                                      [ 4.0888e+00, -3.3949e+01],
                                                      [ 3.4366e+01,  2.7721e+01]],

                                                     [[ 5.2087e+00, -1.3141e+01],
                                                      [-8.3409e+01, -5.3549e+01],
                                                      [ 2.7209e+01, -1.1435e+02],
                                                      [-2.0424e-02,  8.5139e+00]]],


                                                    [[[ 4.6776e+01, -8.4654e-01],
                                                      [-5.5823e+00, -6.0218e+01],
                                                      [ 2.1814e+00,  1.0590e+01],
                                                      [-2.5290e+01,  2.5629e+01]],

                                                     [[ 4.2384e+00, -8.4199e+00],
                                                      [-3.8285e+01,  1.7978e+01],
                                                      [ 2.2481e+01,  6.5141e+01],
                                                      [-7.9511e-01, -9.9825e+00]],

                                                     [[-2.6924e+01, -8.0152e+01],
                                                      [-1.1862e+01,  2.7242e+01],
                                                      [ 3.1457e+01,  4.8352e+01],
                                                      [-8.1167e+01,  3.2597e+01]]]])

            let t3s3p6d3 = t1.conv2d(t2, stride=3, padding=6, dilation=3)
            let t3s3p6d3Correct = combo.tensor([[[[  78.0793,   88.7191,  -32.2774,   12.5512],
                                                      [  27.0241, -107.5002,   98.7433,  -41.9933],
                                                      [  11.7470, -105.7288, -152.6583,   23.1514],
                                                      [ -67.0271,   60.8134,   74.5546,    9.3066]],

                                                     [[  -1.9717,   29.6326,   33.0870,   35.4221],
                                                      [  -3.6938,  -49.7435,  -66.3994,  -25.3134],
                                                      [  35.9503,   38.2935,   80.4125,   -2.5147],
                                                      [  78.7071,  -45.5705,   20.5010,  -15.2868]],

                                                     [[  -9.2327,   96.5872,   28.3565,   92.0639],
                                                      [  35.3198,    5.5638,  -14.6744, -150.4814],
                                                      [ 106.6989, -163.4741,   37.9205,   70.2904],
                                                      [ -62.9899,   25.6233,    7.3010,  -20.2932]]],


                                                    [[[ -41.3512,  -21.4615,   29.8981,   -2.3176],
                                                      [  15.9843,  -22.6151,   87.3233,   36.7436],
                                                      [  46.3618,   66.0061,   18.5348,   38.1597],
                                                      [   6.6860,   65.4270,  -14.5871,  -45.0162]],

                                                     [[ -21.3053,  -12.6932,    4.7727,   -8.6866],
                                                      [ -23.4574,  -39.6679,   -1.5520,  -29.9771],
                                                      [ -66.3903, -127.3519,  -46.1654,  -79.1997],
                                                      [  -4.9704,  -93.0387,  -48.5467,  -39.6767]],

                                                     [[ -26.7460,  -27.8782,  -81.2187,  -76.9048],
                                                      [ -37.5283,  -29.9493,   60.9875,  -86.3384],
                                                      [  26.8834,  -22.3392,   64.3614,   32.6334],
                                                      [  11.1650,   45.6064,   -9.0581,   23.5884]]]])

            let t3b1 = t1.[0].unsqueeze(0).conv2d(t2)
            let t3b1Correct = t3Correct.[0].unsqueeze(0)
            let t3b1s2 = t1.[0].unsqueeze(0).conv2d(t2, stride = 2)
            let t3b1s2Correct = t3s2Correct.[0].unsqueeze(0)

            // Assert.True(false)
            Assert.True(t3.allclose(t3Correct, 0.01))
            Assert.True(t3p1.allclose(t3p1Correct, 0.01))
            Assert.True(t3p12.allclose(t3p12Correct, 0.01))
            Assert.True(t3s2.allclose(t3s2Correct, 0.01))
            Assert.True(t3s13.allclose(t3s13Correct, 0.01))
            Assert.True(t3s2p1.allclose(t3s2p1Correct, 0.01))
            Assert.True(t3s23p32.allclose(t3s23p32Correct, 0.01))
            Assert.True(t3p1d2.allclose(t3p1d2Correct, 0.01))
            Assert.True(t3p22d23.allclose(t3p22d23Correct, 0.01))
            Assert.True(t3s3p6d3.allclose(t3s3p6d3Correct, 0.01))
            Assert.True(t3b1.allclose(t3b1Correct, 0.01))
            Assert.True(t3b1s2.allclose(t3b1s2Correct, 0.01))

        // check intergral types
        for combo in Combos.Integral do 
            let x = combo.ones([1;1;4;4])
            let y = combo.ones([1;1;4;4])
            let z = dsharp.conv2d(x, y)
            let zCorrect = combo.tensor([[[[16]]]])
            Assert.AreEqual(z, zCorrect)

        // check types must always match
        for dtype1 in DTypes.All do 
            for dtype2 in DTypes.All do 
                if dtype1 <> dtype2 then 
                    let x = dsharp.zeros([1;1;4;4], dtype=dtype1)
                    let y = dsharp.zeros([1;1;4;4], dtype=dtype2)
                    isException(fun () -> dsharp.conv2d(x,y, strides=[1;1]))

        for combo in Combos.Bool do 
            let x = combo.zeros([1;1;4;4])
            let y = combo.zeros([1;1;4;4])
            isInvalidOp(fun () -> dsharp.conv2d(x,y, strides=[1;1]))

    [<Test>]
    member _.TestTensorConv3D () =
        for combo in Combos.FloatingPoint do 
        let t1 = combo.tensor([[[[ 2.0403e+00,  5.0188e-01,  4.6880e-01,  8.0736e-01],
                                   [-6.1190e-01,  6.1642e-01, -4.0588e-01, -2.9679e-01],
                                   [-5.6210e-01,  3.6843e-01, -6.6630e-02, -1.3918e+00],
                                   [-1.2988e+00,  9.6719e-01, -3.3539e-01,  8.7715e-01]],

                                  [[-1.7863e+00, -1.1244e+00, -2.1417e-02,  6.4124e-01],
                                   [ 7.5028e-01,  2.2587e-01, -1.2390e-01, -8.4495e-02],
                                   [-1.1291e+00,  1.5644e+00, -2.0280e+00, -9.2168e-01],
                                   [-9.2567e-01,  3.9768e-01,  1.0377e+00,  5.0193e-01]],

                                  [[-5.3238e-01, -8.4971e-02,  5.3398e-01, -1.0695e+00],
                                   [ 5.6227e-01,  2.3256e-01,  6.6780e-01, -7.1462e-01],
                                   [-6.6682e-01, -3.5299e-01, -6.0286e-01, -1.0693e+00],
                                   [ 1.2855e+00, -5.9239e-02, -1.6507e-01, -7.1905e-01]],

                                  [[-4.1638e-01,  7.6894e-01, -8.3663e-01,  8.2333e-01],
                                   [-1.4869e+00, -1.5159e+00,  8.6893e-01, -4.0507e-01],
                                   [ 1.6423e+00,  1.1892e+00,  9.8311e-01, -4.7513e-01],
                                   [ 1.4261e+00, -1.6494e+00,  8.3231e-02,  3.5143e-01]]],


                                 [[[ 1.6732e+00, -2.3141e+00, -2.7201e-01,  4.8099e-02],
                                   [ 1.4185e-01, -2.7953e-01,  2.0087e-01,  2.5665e+00],
                                   [ 2.0306e+00,  1.3222e+00,  2.3076e-01,  4.5952e-01],
                                   [ 8.8091e-01, -7.6203e-01,  1.4536e-03,  1.3817e-01]],

                                  [[-1.8129e-01,  3.7236e-01,  4.3555e-01,  1.0214e+00],
                                   [ 1.7297e-01, -3.5313e-01,  2.8694e+00, -4.7409e-01],
                                   [-6.3609e-01,  3.4134e+00, -4.9251e-01, -3.8600e-01],
                                   [ 6.8581e-02,  1.0088e+00,  3.0463e-01, -5.7993e-01]],

                                  [[ 7.7506e-01,  1.5062e-01, -2.9680e-02, -1.9979e+00],
                                   [ 6.7832e-01,  1.3433e+00,  1.0491e+00,  9.5303e-02],
                                   [-1.4113e+00, -3.0230e-01, -3.2206e-01,  3.3161e-01],
                                   [-1.0122e+00,  5.1443e-01,  6.5048e-02, -4.2270e-02]],

                                  [[ 1.2150e+00, -1.4316e+00, -2.9044e-01, -7.3760e-01],
                                   [ 3.5693e-01,  1.0187e+00,  1.1133e+00, -4.1039e-01],
                                   [-1.7768e+00, -2.2549e-01,  2.7584e-01, -1.2234e+00],
                                   [-2.9351e-01, -5.3639e-01, -1.2375e+00,  8.3979e-03]]]]).unsqueeze(0)
        let t2 = combo.tensor([[[[-0.5868, -0.6268,  0.2067],
                                   [ 0.0902, -0.2625,  0.4332],
                                   [-2.3743,  0.4579,  1.1151]],

                                  [[-0.6703, -0.4771,  1.5989],
                                   [-0.8629,  0.0367, -1.7918],
                                   [-0.1023,  0.0615, -1.3259]],

                                  [[ 0.5963,  0.3167,  0.8568],
                                   [ 1.0630, -0.2076, -1.6126],
                                   [-0.6459,  1.4887, -1.4647]]],


                                 [[[-0.6016,  0.8268,  1.3840],
                                   [-0.2750, -0.2897,  0.9044],
                                   [-1.8141, -0.2568,  0.3517]],

                                  [[ 0.4624, -0.5173, -0.7067],
                                   [-0.3159,  0.7693,  0.0949],
                                   [ 0.2051,  1.2193, -1.5660]],

                                  [[-0.0875,  0.5780, -0.2825],
                                   [ 0.2239,  0.7976,  1.5523],
                                   [ 0.6226, -0.4116,  1.0639]]]]).unsqueeze(0)

        let t3 = t1.conv3d(t2)
        let t3Correct = combo.tensor([[[[ 3.1109,  6.7899],
                                           [ 4.3064,  4.1053]],

                                          [[ 5.0324, -8.8943],
                                           [-0.1298,  1.2862]]]]).unsqueeze(0)

        let t3p1 = t1.conv3d(t2, padding=1)
        let t3p1Correct = combo.tensor([[[[  2.9555,  -2.2637,  -7.1829,   5.6339],
                                           [ -3.3115,  11.7124,   2.7917,   2.6118],
                                           [  5.5319,   3.0030,   3.2099,  -2.7804],
                                           [ -1.4804,  -0.1157,  -6.4439,  -0.0716]],

                                          [[  2.4783,  -2.6479,   5.6216,  -1.2882],
                                           [-10.3388,   3.1109,   6.7899,  -6.1003],
                                           [ -1.3145,   4.3064,   4.1053,   5.3012],
                                           [  2.6878,  -4.5237,  -0.6728,   0.6796]],

                                          [[ -1.4721,  -4.1515,   4.6180,  -9.2384],
                                           [  9.8664,   5.0324,  -8.8943,   5.2075],
                                           [ -1.5404,  -0.1298,   1.2862,  -3.2419],
                                           [  8.5308,   2.7561,  -6.2106,   1.8973]],

                                          [[  0.9938,  -2.9158,  -5.2227,  -3.0340],
                                           [  3.2490,   2.0787,   2.2262,  -2.4861],
                                           [ -0.0842,   0.3416,  -3.8301,  -2.1084],
                                           [  4.0825,  -1.9845,  -1.1269,   2.3267]]]]).unsqueeze(0)

        let t3p123 = t1.conv3d(t2, paddings=[|1; 2; 3|])
        let t3p123Correct = combo.tensor([[[[ 0.0000e+00, -2.9020e+00,  4.5825e+00, -3.1431e+00, -1.0803e+00,
                                                     8.2371e-01,  1.4897e-01,  0.0000e+00],
                                               [ 0.0000e+00, -1.2234e+00,  2.9555e+00, -2.2637e+00, -7.1829e+00,
                                                     5.6339e+00,  5.1473e-01,  0.0000e+00],
                                               [ 0.0000e+00, -6.8862e-01, -3.3115e+00,  1.1712e+01,  2.7917e+00,
                                                     2.6118e+00, -3.8470e-01,  0.0000e+00],
                                               [ 0.0000e+00,  3.3201e+00,  5.5319e+00,  3.0030e+00,  3.2099e+00,
                                                    -2.7804e+00,  6.1979e-01,  0.0000e+00],
                                               [ 0.0000e+00,  8.8853e-01, -1.4804e+00, -1.1566e-01, -6.4439e+00,
                                                    -7.1598e-02,  2.3270e-01,  0.0000e+00],
                                               [ 0.0000e+00, -3.5118e+00,  2.0512e+00,  1.6275e+00,  1.7109e+00,
                                                     1.5145e-01, -1.7395e-01,  0.0000e+00]],

                                              [[ 0.0000e+00,  7.1204e+00,  3.0177e-04, -6.9272e+00,  2.8760e+00,
                                                    -1.9002e-02, -2.4133e+00,  0.0000e+00],
                                               [ 0.0000e+00,  5.6420e+00,  2.4783e+00, -2.6479e+00,  5.6216e+00,
                                                    -1.2882e+00, -5.9195e+00,  0.0000e+00],
                                               [ 0.0000e+00,  7.1537e-02, -1.0339e+01,  3.1109e+00,  6.7899e+00,
                                                    -6.1003e+00,  1.2121e+00,  0.0000e+00],
                                               [ 0.0000e+00,  8.9927e-01, -1.3145e+00,  4.3064e+00,  4.1053e+00,
                                                     5.3012e+00, -4.4293e+00,  0.0000e+00],
                                               [ 0.0000e+00, -5.7960e-01,  2.6878e+00, -4.5237e+00, -6.7276e-01,
                                                     6.7965e-01, -6.6988e-01,  0.0000e+00],
                                               [ 0.0000e+00,  8.0942e-01,  6.4290e-01,  1.2871e+00,  5.3531e-01,
                                                    -1.0901e+00, -1.6275e+00,  0.0000e+00]],

                                              [[ 0.0000e+00, -6.6101e-01, -4.8746e+00,  7.4949e+00,  3.0253e+00,
                                                    -1.3816e+00, -4.6669e+00,  0.0000e+00],
                                               [ 0.0000e+00,  4.2946e+00, -1.4721e+00, -4.1515e+00,  4.6180e+00,
                                                    -9.2384e+00,  3.2005e+00,  0.0000e+00],
                                               [ 0.0000e+00, -2.9133e+00,  9.8664e+00,  5.0324e+00, -8.8943e+00,
                                                     5.2075e+00,  2.1560e+00,  0.0000e+00],
                                               [ 0.0000e+00, -9.4993e+00, -1.5404e+00, -1.2982e-01,  1.2862e+00,
                                                    -3.2419e+00,  4.1770e-01,  0.0000e+00],
                                               [ 0.0000e+00, -4.7673e+00,  8.5308e+00,  2.7561e+00, -6.2106e+00,
                                                     1.8973e+00,  2.6808e+00,  0.0000e+00],
                                               [ 0.0000e+00,  3.9791e+00,  5.8774e-01,  3.1007e-01, -4.0616e+00,
                                                    -8.0652e-01,  7.2560e-01,  0.0000e+00]],

                                              [[ 0.0000e+00, -1.6718e+00,  2.1936e+00,  5.2331e-01, -2.4292e+00,
                                                    -2.0133e+00,  5.9281e+00,  0.0000e+00],
                                               [ 0.0000e+00,  3.6098e+00,  9.9384e-01, -2.9158e+00, -5.2227e+00,
                                                    -3.0340e+00,  1.4565e+00,  0.0000e+00],
                                               [ 0.0000e+00,  2.3582e+00,  3.2490e+00,  2.0787e+00,  2.2262e+00,
                                                    -2.4861e+00,  3.0599e+00,  0.0000e+00],
                                               [ 0.0000e+00, -6.6049e+00, -8.4240e-02,  3.4158e-01, -3.8301e+00,
                                                    -2.1084e+00,  2.8022e+00,  0.0000e+00],
                                               [ 0.0000e+00, -1.1513e+00,  4.0825e+00, -1.9845e+00, -1.1269e+00,
                                                     2.3267e+00, -1.7839e-01,  0.0000e+00],
                                               [ 0.0000e+00,  1.3527e+00, -3.7297e+00,  1.3533e+00,  1.6894e+00,
                                                    -3.2651e-01,  2.1566e-01,  0.0000e+00]]]]).unsqueeze(0)

        let t3s2 = t1.conv3d(t2, stride=2)
        let t3s2Correct = combo.tensor([[[[3.1109]]]]).unsqueeze(0)

        let t3s132 = t1.conv3d(t2, strides=[|1; 3; 2|])
        let t3s132Correct = combo.tensor([[[[3.1109]],
                                              [[5.0324]]]]).unsqueeze(0)

        let t3s2p1 = t1.conv3d(t2, stride=2, padding=1)
        let t3s2p1Correct = combo.tensor([[[[ 2.9555, -7.1829],
                                               [ 5.5319,  3.2099]],

                                              [[-1.4721,  4.6180],
                                               [-1.5404,  1.2862]]]]).unsqueeze(0)

        let t3s231p321 = t1.conv3d(t2, strides=[2; 3; 1], paddings=[3; 2; 1])
        let t3s231p321Correct = combo.tensor([[[[ 0.0000,  0.0000,  0.0000,  0.0000],
                                                   [ 0.0000,  0.0000,  0.0000,  0.0000]],

                                                  [[ 4.5825, -3.1431, -1.0803,  0.8237],
                                                   [ 5.5319,  3.0030,  3.2099, -2.7804]],

                                                  [[-4.8746,  7.4949,  3.0253, -1.3816],
                                                   [-1.5404, -0.1298,  1.2862, -3.2419]],

                                                  [[-0.1487, -1.5309,  1.1215,  3.0797],
                                                   [ 1.4189,  1.4221,  4.1597,  1.4329]]]]).unsqueeze(0)
        
        Assert.True(t3.allclose(t3Correct, 0.01, 0.01))
        Assert.True(t3p1.allclose(t3p1Correct, 0.01, 0.01))
        Assert.True(t3p123.allclose(t3p123Correct, 0.01, 0.01))
        Assert.True(t3s2.allclose(t3s2Correct, 0.01, 0.01))
        Assert.True(t3s132.allclose(t3s132Correct, 0.01, 0.01))
        Assert.True(t3s2p1.allclose(t3s2p1Correct, 0.01, 0.01))
        Assert.True(t3s231p321.allclose(t3s231p321Correct, 0.01, 0.01))

        // 3D dilations not working correctly in LibTorch
        if combo.backend <> Backend.Torch then
            let t3p1d2 = t1.conv3d(t2, padding=1, dilation=2)
            let t3p1d2Correct = combo.tensor([[[[-0.2568,  0.7812],
                                                   [ 3.7157,  2.1968]],

                                                  [[ 7.7515,  1.1481],
                                                   [-1.2951, -2.1536]]]]).unsqueeze(0)
            Assert.True(t3p1d2.allclose(t3p1d2Correct, 0.01, 0.01))

        // 3D dilations not working correctly in LibTorch
        if combo.backend <> Backend.Torch then
            let t3p224d234 = t1.conv3d(t2, paddings=[2;2;4], dilations=[2;3;4])
            let t3p224d234Correct = 
                                   combo.tensor([[[[ 0.5110,  0.8308,  0.8378,  2.1878],
                                                   [ 0.5542,  0.8628,  0.0433,  0.7889]],

                                                  [[ 0.7539,  0.8638,  2.9105, -0.6111],
                                                   [-2.2889,  2.2566, -0.4374, -1.2079]],

                                                  [[ 0.6620,  0.9611,  0.8799, -0.6184],
                                                   [-1.5508, -0.7252, -0.3192,  0.4482]],

                                                  [[-0.0271,  0.7710,  0.0897, -0.1711],
                                                   [-0.8259, -1.5293,  0.9234, -0.6048]]]]).unsqueeze(0)
            Assert.True(t3p224d234.allclose(t3p224d234Correct, 0.01, 0.01))

        // 3D dilations not working correctly in LibTorch
        if combo.backend <> Backend.Torch then
            let t3s3p6d3 = t1.conv3d(t2, stride=3, padding=6, dilation=3)
            let t3s3p6d3Correct = 
                                   combo.tensor([[[[-1.2082,  1.2172,  0.9059, -0.4916],
                                                   [ 2.1467, -3.7502,  5.0506,  0.3885],
                                                   [ 4.7375,  2.0637,  0.0984,  1.4406],
                                                   [-1.3617,  0.8104, -0.4940,  0.5110]],

                                                  [[-3.4229, -2.0909,  2.7974, -1.0638],
                                                   [-2.9979, -0.1444, -3.2004, -0.2850],
                                                   [ 1.0353, -1.1102,  0.8409, -0.3885],
                                                   [-1.3945,  2.0495,  1.7803, -0.3152]],

                                                  [[ 1.5129,  2.9412, -8.0788, -2.2397],
                                                   [ 0.6883, -1.7963,  0.6140, -2.7854],
                                                   [-1.1362,  1.5341, -3.5884, -1.6604],
                                                   [ 3.4384,  1.9425, -1.4670, -0.8295]],

                                                  [[-0.0370,  0.1560, -0.6491, -0.6168],
                                                   [ 2.4056,  0.5702, -3.0690, -0.5726],
                                                   [ 1.9479,  0.2854, -1.4980, -0.0100],
                                                   [-0.1114, -1.0524, -0.8736, -0.2113]]]]).unsqueeze(0)
            Assert.True(t3s3p6d3.allclose(t3s3p6d3Correct, 0.01, 0.01))

    [<Test>]
    member _.TestTensorNegT () =
        // Test all non-bool types
        for combo in Combos.IntegralAndFloatingPoint do 
            let t1 = combo.tensor([1.; 2.; 3.])
            let t1Neg = -t1
            let t1NegCorrect = combo.tensor([-1.; -2.; -3.])

            Assert.AreEqual(t1NegCorrect, t1Neg)
            Assert.AreEqual(t1Neg.dtype, combo.dtype)

        // Neg of Bool tensor not allowed
        //
        //    -torch.ones(10, dtype=torch.bool) 
        //
        // RuntimeError: Negation, the `-` operator, on a bool tensor is not supported. 

        for combo in Combos.Bool do 
            isInvalidOp(fun () -> -combo.tensor([1.0]))

    [<Test>]
    member _.TestTensorSumT () =
        // Test all non-bool types
        for combo in Combos.IntegralAndFloatingPoint do 
            let t1 = combo.tensor([1.; 2.; 3.])
            let t1Sum = t1.sum()
            let t1SumCorrect = combo.tensor(6., dtype=combo.dtype.SummationType)

            Assert.AreEqual(t1Sum.dtype, combo.dtype.SummationType)
            Assert.AreEqual(t1SumCorrect, t1Sum)

            // Now test cases where result type is set explicitly
            for dtype2 in DTypes.IntegralAndFloatingPoint do
                let t1SumTyped = t1.sum(dtype=dtype2)
                let t1SumTypedCorrect = combo.tensor(6., dtype=dtype2)
                Assert.AreEqual(t1SumTyped.dtype, dtype2)
                Assert.AreEqual(t1SumTypedCorrect, t1SumTyped)

            let t2 = combo.tensor([[1.; 2.]; [3.; 4.]])
            let t2Sum = t2.sum()
            let t2SumCorrect = combo.tensor(10., dtype=combo.dtype.SummationType)

            Assert.AreEqual(t2Sum.dtype, combo.dtype.SummationType)
            Assert.AreEqual(t2SumCorrect, t2Sum)

        for combo in Combos.Bool do 
            // Sum of Bool tensor is Int64 tensor in pytorch
            let t3a = combo.tensor([true; true; false])
            let t3 = t3a.sum()
            let t3Correct = combo.tensor(2, dtype=DType.Int64)
            Assert.AreEqual(t3, t3Correct)

    [<Test>]
    member _.TestTensorSumToSizeT () =
        for combo in Combos.IntegralAndFloatingPoint do 
            let t1 = combo.tensor([1.; 2.; 3.])
            let t1Sum = t1.sumToSize([| |])
            let t1SumCorrect = combo.tensor(6., dtype=combo.dtype.SummationType)

            Assert.AreEqual(t1SumCorrect, t1Sum)

            let t2 = combo.tensor([[1.; 2.]; [3.; 4.]])
            let t2Sum = t2.sumToSize([| |])
            let t2SumCorrect = combo.tensor(10., dtype=combo.dtype.SummationType)

            Assert.AreEqual(t2SumCorrect, t2Sum)

            let t3 = combo.tensor([[1.; 2.]; [3.; 4.]])
            let t3Sum = t3.sumToSize([| 2 |])
            let t3SumCorrect = combo.tensor( [4.; 6.], dtype=combo.dtype.SummationType)

            Assert.AreEqual(t3SumCorrect, t3Sum)

            let t4 = combo.tensor([[1.; 2.]; [3.; 4.]])
            let t4Sum = t4.sumToSize([| 1; 2 |])
            let t4SumCorrect = combo.tensor( [ [4.; 6.] ], dtype=combo.dtype.SummationType)

            Assert.AreEqual(t4SumCorrect, t4Sum)

            let t5 = combo.tensor([[1.; 2.]; [3.; 4.]])
            let t5Sum = t5.sumToSize([| 2; 1 |])
            let t5SumCorrect = combo.tensor( [ [3.]; [7.] ], dtype=combo.dtype.SummationType)

            Assert.AreEqual(t5SumCorrect, t5Sum)

    [<Test>]
    member _.TestTensorSumToSizeSystematic () =
        for combo in Combos.IntegralAndFloatingPoint do 
            // Systematically test all legitimate reductions of 2x2x2 to smaller sizes
            let t6 = combo.tensor([ [[1.; 2.]; [3.; 4.] ]; [[5.; 6.]; [7.; 8.] ] ])
            let systematicResults = 
                [| for i1 in 0..2 do 
                      for i2 in (if i1 = 0 then 0 else 1)..2 do
                         for i3 in (if i2 = 0 then 0 else 1)..2 do
                            let newShape = 
                                [| if i1 > 0 then yield i1
                                   if i2 > 0 then yield i2
                                   if i3 > 0 then yield i3 |]
                            yield (newShape, t6.sumToSize(newShape)) |]
        
            let expectedResults = 
                [|([||], combo.tensor (36., dtype=combo.dtype.SummationType));
                  ([|1|], combo.tensor ([36.], dtype=combo.dtype.SummationType));
                  ([|2|], combo.tensor ([16.; 20.], dtype=combo.dtype.SummationType));
                  ([|1; 1|], combo.tensor ([[36.]], dtype=combo.dtype.SummationType));
                  ([|1; 2|], combo.tensor ([[16.; 20.]], dtype=combo.dtype.SummationType));
                  ([|2; 1|], combo.tensor([[14.]; [22.]], dtype=combo.dtype.SummationType));
                  ([|2; 2|], combo.tensor([[6.; 8.]; [10.; 12.]], dtype=combo.dtype.SummationType));
                  ([|1; 1; 1|], combo.tensor([[[36.]]], dtype=combo.dtype.SummationType));
                  ([|1; 1; 2|], combo.tensor([[[16.; 20.]]], dtype=combo.dtype.SummationType));
                  ([|1; 2; 1|], combo.tensor([[[14.]; [22.]]], dtype=combo.dtype.SummationType));
                  ([|1; 2; 2|], combo.tensor([[[6.; 8.]; [10.; 12.]]], dtype=combo.dtype.SummationType));
                  ([|2; 1; 1|], combo.tensor([[[10.]]; [[26.]]], dtype=combo.dtype.SummationType));
                  ([|2; 1; 2|], combo.tensor([[[4.; 6.]]; [[12.; 14.]]], dtype=combo.dtype.SummationType));
                  ([|2; 2; 1|], combo.tensor([[[3.]; [7.]]; [[11.]; [15.]]], dtype=combo.dtype.SummationType));
                  ([|2; 2; 2|], combo.tensor([[[1.; 2.]; [3.; 4.]]; [[5.; 6.]; [7.; 8.]]], dtype=combo.dtype.SummationType))|]

            Assert.AreEqual(systematicResults, expectedResults)

    [<Test>]
    member _.TestTensorSumT2Dim0 () =
        // Test all non-bool types
        for combo in Combos.IntegralAndFloatingPoint do 
            let t1 = combo.tensor([[1.; 2.]; [3.; 4.]])
            let t1Sum = t1.sumT2Dim0()
            let t1SumCorrect = combo.tensor([4.; 6.])

            Assert.AreEqual(t1SumCorrect, t1Sum)
            Assert.AreEqual(t1Sum.dtype, combo.dtype)
    
    [<Test>]
    member _.TestTensorSumDim () =
        // Test all non-bool types
        for combo in Combos.IntegralAndFloatingPoint do 
            let t = combo.tensor([[[1.,2.,3.,4.], [5.,6.,7.,8.], [9.,10.,11.,12.]], [[13.,14.,15.,16.], [17.,18.,19.,20.], [21.,22.,23.,24.]]])
            let tSum0 = t.sum(0)
            let tSum0Correct = combo.tensor([[14.0f, 16.0f, 18.0f, 20.0f], [22.0f, 24.0f, 26.0f, 28.0f], [30.0f, 32.0f, 34.0f, 36.0f]], dtype=combo.dtype.SummationType)
            let tSum1 = t.sum(1)
            let tSum1Correct = combo.tensor([[15.0f, 18.0f, 21.0f, 24.0f], [51.0f, 54.0f, 57.0f, 60.0f]], dtype=combo.dtype.SummationType)
            let tSum2 = t.sum(2)
            let tSum2Correct = combo.tensor([[10.0f, 26.0f, 42.0f], [58.0f, 74.0f, 90.0f]], dtype=combo.dtype.SummationType)

            Assert.AreEqual(tSum0.dtype, combo.dtype.SummationType)
            Assert.AreEqual(tSum1.dtype, combo.dtype.SummationType)
            Assert.AreEqual(tSum2.dtype, combo.dtype.SummationType)
            Assert.AreEqual(tSum0Correct, tSum0)
            Assert.AreEqual(tSum1Correct, tSum1)
            Assert.AreEqual(tSum2Correct, tSum2)
    
    [<Test>]
    member _.TestTensorSumDimKeepDim () =
        // Test all non-bool types
        for combo in Combos.IntegralAndFloatingPoint do 
            let t = combo.tensor([[[1.;2.;3.;4.]; [5.;6.;7.;8.]; [9.;10.;11.;12.]]; [[13.;14.;15.;16.]; [17.;18.;19.;20.]; [21.;22.;23.;24.]]])
            let tSum0 = t.sum(0, keepDim=true)
            let tSum0Correct = combo.tensor([[[14.0f; 16.0f; 18.0f; 20.0f]; [22.0f; 24.0f; 26.0f; 28.0f]; [30.0f; 32.0f; 34.0f; 36.0f]]], dtype=combo.dtype.SummationType)
            let tSum1 = t.sum(1, keepDim=true)
            let tSum1Correct = combo.tensor([[[15.0f; 18.0f; 21.0f; 24.0f]]; [[51.0f; 54.0f; 57.0f; 60.0f]]], dtype=combo.dtype.SummationType)
            let tSum2 = t.sum(2, keepDim=true)
            let tSum2Correct = combo.tensor([[[10.0f]; [26.0f]; [42.0f]]; [[58.0f]; [74.0f]; [90.0f]]], dtype=combo.dtype.SummationType)

            Assert.AreEqual(tSum0.dtype, combo.dtype.SummationType)
            Assert.AreEqual(tSum1.dtype, combo.dtype.SummationType)
            Assert.AreEqual(tSum2.dtype, combo.dtype.SummationType)
            Assert.AreEqual(tSum0Correct, tSum0)
            Assert.AreEqual(tSum1Correct, tSum1)
            Assert.AreEqual(tSum2Correct, tSum2)

    [<Test>]
    member _.TestTensorMean () =
        for combo in Combos.FloatingPoint do 
            let t = combo.tensor([[[1.;2.;3.;4.]; [5.;6.;7.;8.]; [9.;10.;11.;12.]]; [[13.;14.;15.;16.]; [17.;18.;19.;20.]; [21.;22.;23.;24.]]])
            let tMean = t.mean()
            let tMeanCorrect = combo.tensor(12.5)

            Assert.AreEqual(tMeanCorrect, tMean)
            Assert.AreEqual(tMean.dtype, combo.dtype)

            // mean, dim={0,1,2}
            (* Python:
            import pytorch as torch
            input = np.[[[1.,2.,3.,4.], [5.,6.,7.,8.], [9.,10.,11.,12.]], [[13.,14.,15.,16.], [17.,18.,19.,20.], [21.,22.,23.,24.]]]
            input.mean(1)
            --> array([[15., 18., 21., 24.],[51., 54., 57., 60.]])
            input.sum(2)
            --> array([[10., 26., 42.],[58., 74., 90.]])
            *)
            let tMean0 = t.mean(0)
            let tMean0Correct = combo.tensor([[7.; 8.; 9.; 10.]; [11.; 12.; 13.; 14.]; [15.; 16.; 17.; 18.]])
            let tMean1 = t.mean(1)
            let tMean1Correct = combo.tensor([[5.; 6.; 7.; 8.]; [17.; 18.; 19.; 20.]])
            let tMean2 = t.mean(2)
            let tMean2Correct = combo.tensor([[2.5; 6.5; 10.5]; [14.5; 18.5; 22.5]])

            Assert.AreEqual(tMean0Correct, tMean0)
            Assert.AreEqual(tMean1Correct, tMean1)
            Assert.AreEqual(tMean2Correct, tMean2)

            // mean, dim={0,1,2}, keepDim=true
            (* Python:
            import torch
            input = torch.tensor([[[1.,2.,3.,4.], [5.,6.,7.,8.], [9.,10.,11.,12.]], [[13.,14.,15.,16.], [17.,18.,19.,20.], [21.,22.,23.,24.]]])
            input.mean(0,keepdim=True)
            # --> tensor([[[ 7.,  8.,  9., 10.],[11., 12., 13., 14.],[15., 16., 17., 18.]]])
            input.mean(1,keepdim=True)
            # --> tensor([[[ 5.,  6.,  7.,  8.]],[[17., 18., 19., 20.]]])
            input.mean(2,keepdim=True)
            # --> tensor([[[ 2.5000],[ 6.5000],[10.5000]],[[14.5000],[18.5000],[22.5000]]])
            *)
            let tMeanKeepDim0 = t.mean(0, keepDim=true)
            let tMeanKeepDim0Correct = combo.tensor([[[7.; 8.; 9.; 10.]; [11.; 12.; 13.; 14.]; [15.; 16.; 17.; 18.]]])
            let tMeanKeepDim1 = t.mean(1, keepDim=true)
            let tMeanKeepDim1Correct = combo.tensor([[[5.; 6.; 7.; 8.]]; [[17.; 18.; 19.; 20.]]])
            let tMeanKeepDim2 = t.mean(2, keepDim=true)
            let tMeanKeepDim2Correct = combo.tensor([[[2.5]; [6.5]; [10.5]]; [[14.5]; [18.5]; [22.5]]])

            Assert.AreEqual(tMeanKeepDim0, tMeanKeepDim0Correct)
            Assert.AreEqual(tMeanKeepDim1, tMeanKeepDim1Correct)
            Assert.AreEqual(tMeanKeepDim2, tMeanKeepDim2Correct)

    [<Test>]
    member _.TestTensorStddev () =
        for combo in Combos.FloatingPoint do 
            let t = combo.tensor([[[0.3787;0.7515;0.2252;0.3416];
                [0.6078;0.4742;0.7844;0.0967];
                [0.1416;0.1559;0.6452;0.1417]];
 
                [[0.0848;0.4156;0.5542;0.4166];
                [0.5187;0.0520;0.4763;0.1509];
                [0.4767;0.8096;0.1729;0.6671]]])
            let tStddev = t.stddev()
            let tStddevCorrect = combo.tensor(0.2398)

            Assert.True(tStddev.allclose(tStddevCorrect, 0.01))
            Assert.AreEqual(tStddev.dtype, combo.dtype)

            // stddev, dim={0,1,2,3}, keepDim=true
            let tStddev0 = t.stddev(0)
            let tStddev0Correct = combo.tensor([[0.2078; 0.2375; 0.2326; 0.0530];
                [0.0630; 0.2985; 0.2179; 0.0383];
                [0.2370; 0.4623; 0.3339; 0.3715]])
            let tStddev1 = t.stddev(1)
            let tStddev1Correct = combo.tensor([[0.2331; 0.2981; 0.2911; 0.1304];
                [0.2393; 0.3789; 0.2014; 0.2581]])
            let tStddev2 = t.stddev(2)
            let tStddev2Correct = combo.tensor([[0.2277; 0.2918; 0.2495];[0.1996; 0.2328; 0.2753]])

            Assert.True(tStddev0.allclose(tStddev0Correct, 0.01))
            Assert.True(tStddev1.allclose(tStddev1Correct, 0.01))
            Assert.True(tStddev2.allclose(tStddev2Correct, 0.01))
            Assert.AreEqual(tStddev0.dtype, combo.dtype)
            Assert.AreEqual(tStddev1.dtype, combo.dtype)
            Assert.AreEqual(tStddev2.dtype, combo.dtype)

            // stddev, dim={0,1,2,3}, keepDim=true
            (* Python:
            import torch
            input = torch.tensor([[[0.3787,0.7515,0.2252,0.3416],[0.6078,0.4742,0.7844,0.0967],[0.1416,0.1559,0.6452,0.1417]],[[0.0848,0.4156,0.5542,0.4166],[0.5187,0.0520,0.4763,0.1509],[0.4767,0.8096,0.1729,0.6671]]])
            input.std(0,keepdim=True)
            # --> tensor([[[0.2078, 0.2375, 0.2326, 0.0530],[0.0630, 0.2985, 0.2179, 0.0383],[0.2370, 0.4622, 0.3340, 0.3715]]])
            input.std(1,keepdim=True)
            # --> tensor([[[0.2331, 0.2980, 0.2911, 0.1304]],[[0.2393, 0.3789, 0.2015, 0.2581]]])
            input.std(2,keepdim=True)
            # --> tensor([[[0.2278],[0.2918],[0.2495]],[[0.1996],[0.2328],[0.2753]]]) 
            *)
            let tStddev0 = t.stddev(0, keepDim=true)
            let tStddev0Correct = combo.tensor([[[0.2078; 0.2375; 0.2326; 0.0530];[0.0630; 0.2985; 0.2179; 0.0383];[0.2370; 0.4623; 0.3339; 0.3715]]])
            let tStddev1 = t.stddev(1, keepDim=true)
            let tStddev1Correct = combo.tensor([[[0.2331; 0.2981; 0.2911; 0.1304]];[[0.2393; 0.3789; 0.2014; 0.2581]]])
            let tStddev2 = t.stddev(2, keepDim=true)
            let tStddev2Correct = combo.tensor([[[0.2277]; [0.2918]; [0.2495]];[[0.1996]; [0.2328]; [0.2753]]])

            Assert.True(tStddev0.allclose(tStddev0Correct, 0.01))
            Assert.True(tStddev1.allclose(tStddev1Correct, 0.01))
            Assert.True(tStddev2.allclose(tStddev2Correct, 0.01))

    [<Test>]
    member _.TestTensorVariance () =
        for combo in Combos.FloatingPoint do 
            (* Python:
            import torch
            input = torch.tensor([[[0.3787,0.7515,0.2252,0.3416],[0.6078,0.4742,0.7844,0.0967],[0.1416,0.1559,0.6452,0.1417]],[[0.0848,0.4156,0.5542,0.4166],[0.5187,0.0520,0.4763,0.1509],[0.4767,0.8096,0.1729,0.6671]]])
            input.var()
            *)
            let t = combo.tensor([[[0.3787;0.7515;0.2252;0.3416]; [0.6078;0.4742;0.7844;0.0967]; [0.1416;0.1559;0.6452;0.1417]]; [[0.0848;0.4156;0.5542;0.4166];[0.5187;0.0520;0.4763;0.1509];[0.4767;0.8096;0.1729;0.6671]]])
            let tVariance = t.variance()
            let tVarianceCorrect = combo.tensor(0.0575)

            Assert.True(tVariance.allclose(tVarianceCorrect, 0.01))

            // Variance, dim={0,1,2,3}
            (* Python:
            input.var(0)
            # --> tensor([[0.0432, 0.0564, 0.0541, 0.0028],[0.0040, 0.0891, 0.0475, 0.0015],[0.0561, 0.2137, 0.1115, 0.1380]])
            input.var(1)
            # --> tensor([[0.0543, 0.0888, 0.0847, 0.0170],[0.0573, 0.1436, 0.0406, 0.0666]])
            input.var(2)
            # --> tensor([[0.0519, 0.0852, 0.0622],[0.0398, 0.0542, 0.0758]])
            *)
            let tVariance0 = t.variance(0)
            let tVariance0Correct = combo.tensor([[0.0432; 0.0564; 0.0541; 0.0028];[0.0040; 0.0891; 0.0475; 0.0015];[0.0561; 0.2137; 0.1115; 0.1380]])
            let tVariance1 = t.variance(1)
            let tVariance1Correct = combo.tensor([[0.0543; 0.0888; 0.0847; 0.0170];[0.0573; 0.1436; 0.0406; 0.0666]])
            let tVariance2 = t.variance(2)
            let tVariance2Correct = combo.tensor([[0.0519; 0.0852; 0.0622];[0.0398; 0.0542; 0.0758]])

            Assert.True(tVariance0.allclose(tVariance0Correct, 0.01, 0.01))
            Assert.True(tVariance1.allclose(tVariance1Correct, 0.01, 0.01))
            Assert.True(tVariance2.allclose(tVariance2Correct, 0.01, 0.01))
            Assert.AreEqual(tVariance0.dtype, combo.dtype)
            Assert.AreEqual(tVariance1.dtype, combo.dtype)
            Assert.AreEqual(tVariance2.dtype, combo.dtype)

    [<Test>]
    member _.TestTensorVarianceKeepDim () =
        for combo in Combos.FloatingPoint do 
            // Variance, dim={0,1,2,3}, keepDim=true
            (* Python:
            import torch
            input = torch.tensor([[[0.3787,0.7515,0.2252,0.3416],[0.6078,0.4742,0.7844,0.0967],[0.1416,0.1559,0.6452,0.1417]],[[0.0848,0.4156,0.5542,0.4166],[0.5187,0.0520,0.4763,0.1509],[0.4767,0.8096,0.1729,0.6671]]])
            input.var(0,keepdim=True)
            # --> tensor([[[0.0432, 0.0564, 0.0541, 0.0028],[0.0040, 0.0891, 0.0475, 0.0015],[0.0561, 0.2137, 0.1115, 0.1380]]])
            input.var(1,keepdim=True)
            # --> tensor([[[0.0543, 0.0888, 0.0847, 0.0170]],[[0.0573, 0.1436, 0.0406, 0.0666]]])
            input.var(2,keepdim=True)
            # --> tensor([[[0.0519],[0.0852],[0.0622]],[[0.0398],[0.0542],[0.0758]]])
            *)
            let t = combo.tensor([[[0.3787;0.7515;0.2252;0.3416]; [0.6078;0.4742;0.7844;0.0967]; [0.1416;0.1559;0.6452;0.1417]]; [[0.0848;0.4156;0.5542;0.4166];[0.5187;0.0520;0.4763;0.1509];[0.4767;0.8096;0.1729;0.6671]]])
            let tVariance0 = t.variance(0, keepDim=true)
            let tVariance0Correct = combo.tensor([[[0.0432; 0.0564; 0.0541; 0.0028];[0.0040; 0.0891; 0.0475; 0.0015];[0.0561; 0.2137; 0.1115; 0.1380]]])
            let tVariance1 = t.variance(1, keepDim=true)
            let tVariance1Correct = combo.tensor([[[0.0543; 0.0888; 0.0847; 0.0170]];[[0.0573; 0.1436; 0.0406; 0.0666]]])
            let tVariance2 = t.variance(2, keepDim=true)
            let tVariance2Correct = combo.tensor([[[0.0519];[0.0852];[0.0622]];[[0.0398];[0.0542];[0.0758]]])

            Assert.True(tVariance0.allclose(tVariance0Correct, 0.01, 0.01))
            Assert.True(tVariance1.allclose(tVariance1Correct, 0.01, 0.01))
            Assert.True(tVariance2.allclose(tVariance2Correct, 0.01, 0.01))
            Assert.AreEqual(tVariance0.dtype, combo.dtype)
            Assert.AreEqual(tVariance1.dtype, combo.dtype)
            Assert.AreEqual(tVariance2.dtype, combo.dtype)

    [<Test>]
    member _.TestTensorTransposeT2 () =
        for combo in Combos.All do 
            let t1 = combo.tensor([[1.; 2.; 3.]; [4.; 5.; 6.]])
            let t1Transpose = t1.transpose()
            let t1TransposeCorrect = combo.tensor([[1.; 4.]; [2.; 5.]; [3.; 6.]])

            let t2 = combo.tensor([[1.; 2.]; [3.; 4.]])
            let t2TransposeTranspose = t2.transpose().transpose()
            let t2TransposeTransposeCorrect = t2

            Assert.AreEqual(t1TransposeCorrect, t1Transpose)
            Assert.AreEqual(t2TransposeTransposeCorrect, t2TransposeTranspose)
            Assert.AreEqual(t1Transpose.dtype, combo.dtype)
            Assert.AreEqual(t2TransposeTranspose.dtype, combo.dtype)

    [<Test>]
    member _.TestTensorSignT () =
        // Test all signed types
        for combo in Combos.SignedIntegralAndFloatingPoint do 
            let t1 = combo.tensor([-1.; -2.; 0.; 3.])
            let t1Sign = t1.sign()
            let t1SignCorrect = combo.tensor([-1.; -1.; 0.; 1.])

            Assert.AreEqual(t1SignCorrect, t1Sign)
            Assert.AreEqual(t1Sign.dtype, combo.dtype)

        // Test all signed types
        for combo in Combos.UnsignedIntegral do 
            let t1 = combo.tensor([1; 1; 0; 3])
            let t1Sign = t1.sign()
            let t1SignCorrect = combo.tensor([1; 1; 0; 1])

            Assert.AreEqual(t1SignCorrect, t1Sign)
            Assert.AreEqual(t1Sign.dtype, combo.dtype)

        // Test bool type separately
        // Note, PyTorch 'torch.tensor([True, False]).sign()' gives 'tensor([ True, False])'
        for combo in Combos.AllDevicesAndBackends do
            let t1Bool = combo.tensor([true;false], dtype=DType.Bool)
            let t1BoolSignCorrect = combo.tensor([true; false], dtype=DType.Bool)

            Assert.AreEqual(t1BoolSignCorrect, t1Bool.sign())

    [<Test>]
    member _.TestTensorFloorT () =
        for combo in Combos.FloatingPoint do 
            let t1 = combo.tensor([0.9473; 0.4891; 0.2015; 0.5818; 0.8439])
            let t1Floor = t1.floor()
            let t1FloorCorrect = combo.tensor([0.; 0.; 0.; 0.; 0.])

            Assert.True(t1Floor.allclose(t1FloorCorrect, 0.01))
            Assert.AreEqual(t1Floor.dtype, combo.dtype)

        for combo in Combos.IntegralAndBool do
            isInvalidOp(fun () -> combo.tensor([1.0]).floor())

    [<Test>]
    member _.TestTensorCeilT () =
        for combo in Combos.FloatingPoint do 
            let t1 = combo.tensor([0.9473; 0.4891; 0.2015; 0.5818; 0.8439])
            let t1Ceil = t1.ceil()
            let t1CeilCorrect = combo.tensor([1.; 1.; 1.; 1.; 1.])

            Assert.True(t1Ceil.allclose(t1CeilCorrect, 0.01))
            Assert.AreEqual(t1Ceil.dtype, combo.dtype)

        for combo in Combos.IntegralAndBool do
            isInvalidOp(fun () -> combo.tensor([1.0]).ceil())

    [<Test>]
    member _.TestTensorRoundT () =
        for combo in Combos.FloatingPoint do 
            let t1 = combo.tensor([0.9473; 0.4891; 0.2015; 0.5818; 0.8439])
            let t1Round = t1.round()
            let t1RoundCorrect = combo.tensor([1.; 0.; 0.; 1.; 1.])

            Assert.True(t1Round.allclose(t1RoundCorrect, 0.01))
            Assert.AreEqual(t1Round.dtype, combo.dtype)

        for combo in Combos.IntegralAndBool do
            isInvalidOp(fun () -> combo.tensor([1.0]).round())

    [<Test>]
    member _.TestTensorAbsT () =
        for combo in Combos.SignedIntegralAndFloatingPoint do 
            let t1 = combo.tensor([-1.; -2.; 0.; 3.])
            let t1Abs = t1.abs()
            let t1AbsCorrect = combo.tensor([1.; 2.; 0.; 3.])

            Assert.AreEqual(t1AbsCorrect, t1Abs)
            Assert.AreEqual(t1Abs.dtype, combo.dtype)

        for combo in Combos.UnsignedIntegral do 
            let t1 = combo.tensor([1.; 2.; 0.; 3.])
            let t1Abs = t1.abs()
            let t1AbsCorrect = combo.tensor([1.; 2.; 0.; 3.])

            Assert.AreEqual(t1AbsCorrect, t1Abs)
            Assert.AreEqual(t1Abs.dtype, combo.dtype)

        // Test bool separately
        // Note: PyTorch fails on 'torch.tensor([True, False]).abs()'
        for combo in Combos.AllDevicesAndBackends do
            let t1 = combo.tensor([true; false], dtype=DType.Bool)
            isInvalidOp (fun () -> t1.abs())

    [<Test>]
    member _.TestTensorReluT () =
        for combo in Combos.SignedIntegralAndFloatingPoint do 
            let t1 = combo.tensor([-1.; -2.; 0.; 3.; 10.])
            let t1Relu = t1.relu()
            let t1ReluCorrect = combo.tensor([0.; 0.; 0.; 3.; 10.])

            Assert.AreEqual(t1ReluCorrect, t1Relu)
            Assert.AreEqual(t1Relu.dtype, combo.dtype)

        // Test bool separately
        for combo in Combos.AllDevicesAndBackends do
            let t1 = combo.tensor([true; false], dtype=DType.Bool)
            isInvalidOp (fun () -> t1.relu())

    [<Test>]
    member _.TestTensorLeakyRelu () =
        for combo in Combos.FloatingPoint do 
            let t1 = combo.tensor([-1.; -2.; 0.; 3.; 10.])
            let t1LeakyRelu = t1.leakyRelu()
            let t1LeakyReluCorrect = combo.tensor([-1.0000e-02; -2.0000e-02;  0.0000e+00;  3.0000e+00;  1.0000e+01])

            Assert.AreEqual(t1LeakyReluCorrect, t1LeakyRelu)
            Assert.AreEqual(t1LeakyRelu.dtype, combo.dtype)
            Assert.AreEqual(t1LeakyRelu.dtype, combo.dtype)

    [<Test>]
    member _.TestTensorSigmoidT () =
        for combo in Combos.FloatingPoint do 
            let t1 = combo.tensor([0.9473; 0.4891; 0.2015; 0.5818; 0.8439])
            let t1Sigmoid = t1.sigmoid()
            let t1SigmoidCorrect = combo.tensor([0.7206; 0.6199; 0.5502; 0.6415; 0.6993])

            Assert.True(t1Sigmoid.allclose(t1SigmoidCorrect, 0.01))
            Assert.AreEqual(t1Sigmoid.dtype, combo.dtype)

        for combo in Combos.IntegralAndBool do
          isInvalidOp(fun () -> combo.tensor([1.0]).sigmoid())

    [<Test>]
    member _.TestTensorSoftplusT () =
        for combo in Combos.FloatingPoint do 
            let t1 = combo.tensor([-1.9908e-01,  9.0179e-01, -5.7899e-01,  1.2083e+00, -4.0689e+04, 2.8907e+05, -6.5848e+05, -1.2992e+05])
            let t1Softplus = t1.softplus()
            let t1SoftplusCorrect = combo.tensor([5.9855e-01, 1.2424e+00, 4.4498e-01, 1.4697e+00, 0.0000e+00, 2.8907e+05, 0.0000e+00, 0.0000e+00])

            Assert.True(t1Softplus.allclose(t1SoftplusCorrect, 0.01))
            Assert.AreEqual(t1Softplus.dtype, combo.dtype)

        for combo in Combos.IntegralAndBool do
            isInvalidOp(fun () -> combo.tensor([1.0]).softplus())

    [<Test>]
    member _.TestTensorExpT () =
        for combo in Combos.FloatingPoint do 
            let t1 = combo.tensor([0.9139; -0.5907;  1.9422; -0.7763; -0.3274])
            let t1Exp = t1.exp()
            let t1ExpCorrect = combo.tensor([2.4940; 0.5539; 6.9742; 0.4601; 0.7208])

            Assert.True(t1Exp.allclose(t1ExpCorrect, 0.01))
            Assert.AreEqual(t1Exp.dtype, combo.dtype)

        for combo in Combos.IntegralAndBool do
            isInvalidOp(fun () -> combo.tensor([1.0]).exp())

    [<Test>]
    member _.TestTensorLogT () =
        for combo in Combos.FloatingPoint do 
            let t1 = combo.tensor([0.1285; 0.5812; 0.6505; 0.3781; 0.4025])
            let t1Log = t1.log()
            let t1LogCorrect = combo.tensor([-2.0516; -0.5426; -0.4301; -0.9727; -0.9100])

            Assert.True(t1Log.allclose(t1LogCorrect, 0.01))
            Assert.AreEqual(t1Log.dtype, combo.dtype)

        for combo in Combos.IntegralAndBool do
            isInvalidOp(fun () -> combo.tensor([1.0]).log())

    [<Test>]
    member _.TestTensorLog10T () =
        for combo in Combos.FloatingPoint do 
            let t1 = combo.tensor([0.1285; 0.5812; 0.6505; 0.3781; 0.4025])
            let t1Log10 = t1.log10()
            let t1Log10Correct = combo.tensor([-0.8911; -0.2357; -0.1868; -0.4224; -0.3952])

            Assert.True(t1Log10.allclose(t1Log10Correct, 0.01))
            Assert.AreEqual(t1Log10.dtype, combo.dtype)

        for combo in Combos.IntegralAndBool do
            isInvalidOp(fun () -> combo.tensor([1.0]).log10())

    [<Test>]
    member _.TestTensorSqrtT () =
        for combo in Combos.FloatingPoint do 
            let t1 = combo.tensor([54.7919; 70.6440; 16.0868; 74.5486; 82.9318])
            let t1Sqrt = t1.sqrt()
            let t1SqrtCorrect = combo.tensor([7.4022; 8.4050; 4.0108; 8.6342; 9.1067])

            Assert.True(t1Sqrt.allclose(t1SqrtCorrect, 0.01))
            Assert.AreEqual(t1Sqrt.dtype, combo.dtype)

        for combo in Combos.IntegralAndBool do
            isInvalidOp(fun () -> combo.tensor([1.0]).sqrt())

    [<Test>]
    member _.TestTensorSinT () =
        for combo in Combos.FloatingPoint do 
            let t1 = combo.tensor([54.7919; 70.6440; 16.0868; 74.5486; 82.9318])
            let t1Sin = t1.sin()
            let t1SinCorrect = combo.tensor([-0.9828;  0.9991; -0.3698; -0.7510;  0.9491])

            Assert.True(t1Sin.allclose(t1SinCorrect, 0.01))
            Assert.AreEqual(t1Sin.dtype, combo.dtype)

        for combo in Combos.IntegralAndBool do
            isInvalidOp(fun () -> combo.tensor([1.0]).sin())

    [<Test>]
    member _.TestTensorCosT () =
        for combo in Combos.FloatingPoint do 
            let t1 = combo.tensor([54.7919; 70.6440; 16.0868; 74.5486; 82.9318])
            let t1Cos = t1.cos()
            let t1CosCorrect = combo.tensor([-0.1849;  0.0418; -0.9291;  0.6603;  0.3150])

            Assert.True(t1Cos.allclose(t1CosCorrect, 0.01))
            Assert.AreEqual(t1Cos.dtype, combo.dtype)

        for combo in Combos.IntegralAndBool do
            isInvalidOp(fun () -> combo.tensor([1.0]).cos())

    [<Test>]
    member _.TestTensorTanT () =
        for combo in Combos.FloatingPoint do 
            let t1 = combo.tensor([0.9473; 1.4891; 0.2015; 0.5818; 0.8439])
            let t1Tan = t1.tan()
            let t1TanCorrect = combo.tensor([1.3904; 12.2132;  0.2043;  0.6577;  1.1244])

            Assert.True(t1Tan.allclose(t1TanCorrect, 0.01))
            Assert.AreEqual(t1Tan.dtype, combo.dtype)

        for combo in Combos.IntegralAndBool do
            isInvalidOp(fun () -> combo.tensor([1.0]).tan())

    [<Test>]
    member _.TestTensorSinhT () =
        for combo in Combos.FloatingPoint do 
            let t1 = combo.tensor([0.9473; 1.4891; 0.2015; 0.5818; 0.8439])
            let t1Sinh = t1.sinh()
            let t1SinhCorrect = combo.tensor([1.0955; 2.1038; 0.2029; 0.6152; 0.9477])

            Assert.True(t1Sinh.allclose(t1SinhCorrect, 0.01))
            Assert.AreEqual(t1Sinh.dtype, combo.dtype)

        for combo in Combos.IntegralAndBool do
            isInvalidOp(fun () -> combo.tensor([1.0]).sinh())

    [<Test>]
    member _.TestTensorCoshT () =
        for combo in Combos.FloatingPoint do 
            let t1 = combo.tensor([0.9473; 1.4891; 0.2015; 0.5818; 0.8439])
            let t1Cosh = t1.cosh()
            let t1CoshCorrect = combo.tensor([1.4833; 2.3293; 1.0204; 1.1741; 1.3777])

            Assert.True(t1Cosh.allclose(t1CoshCorrect, 0.01))
            Assert.AreEqual(t1Cosh.dtype, combo.dtype)

        for combo in Combos.IntegralAndBool do
            isInvalidOp(fun () -> combo.tensor([1.0]).cosh())

    [<Test>]
    member _.TestTensorTanhT () =
        for combo in Combos.FloatingPoint do 
            let t1 = combo.tensor([0.9473; 1.4891; 0.2015; 0.5818; 0.8439])
            let t1Tanh = t1.tanh()
            let t1TanhCorrect = combo.tensor([0.7386; 0.9032; 0.1988; 0.5240; 0.6879])

            Assert.True(t1Tanh.allclose(t1TanhCorrect, 0.01))
            Assert.AreEqual(t1Tanh.dtype, combo.dtype)

        for combo in Combos.IntegralAndBool do
            isInvalidOp(fun () -> combo.tensor([1.0]).tanh())

    [<Test>]
    member _.TestTensorAsinT () =
        for combo in Combos.FloatingPoint do 
            let t1 = combo.tensor([0.9473; 0.4891; 0.2015; 0.5818; 0.8439])
            let t1Asin = t1.asin()
            let t1AsinCorrect = combo.tensor([1.2447; 0.5111; 0.2029; 0.6209; 1.0045])

            Assert.True(t1Asin.allclose(t1AsinCorrect, 0.01))
            Assert.AreEqual(t1Asin.dtype, combo.dtype)

        for combo in Combos.IntegralAndBool do
            isInvalidOp(fun () -> combo.tensor([1.0]).asin())

    [<Test>]
    member _.TestTensorAcosT () =
        for combo in Combos.FloatingPoint do 
            let t1 = combo.tensor([0.9473; 0.4891; 0.2015; 0.5818; 0.8439])
            let t1Acos = t1.acos()
            let t1AcosCorrect = combo.tensor([0.3261; 1.0597; 1.3679; 0.9499; 0.5663])

            Assert.True(t1Acos.allclose(t1AcosCorrect, 0.01))
            Assert.AreEqual(t1Acos.dtype, combo.dtype)

        for combo in Combos.IntegralAndBool do
            isInvalidOp(fun () -> combo.tensor([1.0]).acos())

    [<Test>]
    member _.TestTensorAtanT () =
        for combo in Combos.FloatingPoint do 
            let t1 = combo.tensor([0.9473; 0.4891; 0.2015; 0.5818; 0.8439])
            let t1Atan = t1.atan()
            let t1AtanCorrect = combo.tensor([0.7583; 0.4549; 0.1988; 0.5269; 0.7009])

            Assert.True(t1Atan.allclose(t1AtanCorrect, 0.01))
            Assert.AreEqual(t1Atan.dtype, combo.dtype)

        for combo in Combos.IntegralAndBool do
            isInvalidOp(fun () -> combo.tensor([1.0]).atan())

    [<Test>]
    member _.TestTensorSlice () =
        for combo in Combos.All do 
            let t1 = combo.tensor([1.;2.])
            let t1s1 = t1.[0]
            let t1s2 = t1.[*]
            let t1s1Correct = combo.tensor(1.)
            let t1s2Correct = combo.tensor([1.;2.])

            let t2 = combo.tensor([[1.;2.];[3.;4.]])
            let t2s1 = t2.[0]
            let t2s2 = t2.[*]
            let t2s3 = t2.[0,0]
            let t2s4 = t2.[0,*]
            let t2s5 = t2.[*,0]
            let t2s6 = t2.[*,*]
            let t2s1Correct = combo.tensor([1.;2.])
            let t2s2Correct = combo.tensor([[1.;2.];[3.;4.]])
            let t2s3Correct = combo.tensor(1.)
            let t2s4Correct = combo.tensor([1.;2.])
            let t2s5Correct = combo.tensor([1.;3.])
            let t2s6Correct = combo.tensor([[1.;2.];[3.;4.]])

            let t2b = combo.tensor([[1.;2.;3.;4.]; [5.;6.;7.;8.]; [9.;10.;11.;12.]])
            let t2bs1 = t2b.[1..,2..]
            let t2bs1Correct = combo.tensor([[7.;8.];[11.;12.]])
            let t2bs2 = t2b.[1..2,2..3]
            let t2bs2Correct = combo.tensor([[7.;8.];[11.;12.]])

            let t3 = combo.tensor([[[1.;2.];[3.;4.]];[[5.;6.];[7.;8.]]])
            let t3s1  = t3.[0]
            let t3s2  = t3.[*]
            let t3s3  = t3.[0,0]
            let t3s4  = t3.[0,*]
            let t3s5  = t3.[*,0]
            let t3s6  = t3.[*,*]
            let t3s7  = t3.[0,0,0]
            let t3s8  = t3.[0,0,*]
            let t3s9  = t3.[0,*,0]
            let t3s10 = t3.[0,*,*]
            let t3s11 = t3.[*,0,0]
            let t3s12 = t3.[*,0,*]
            let t3s13 = t3.[*,*,0]
            let t3s14 = t3.[*,*,*]
            let t3s1Correct  = combo.tensor([[1.;2.];[3.;4.]])
            let t3s2Correct  = combo.tensor([[[1.;2.];[3.;4.]];[[5.;6.];[7.;8.]]])
            let t3s3Correct  = combo.tensor([1.;2.])
            let t3s4Correct  = combo.tensor([[1.;2.];[3.;4.]])
            let t3s5Correct  = combo.tensor([[1.;2.];[5.;6.]])
            let t3s6Correct  = combo.tensor([[[1.;2.];[3.;4.]];[[5.;6.];[7.;8.]]])
            let t3s7Correct  = combo.tensor(1.)
            let t3s8Correct  = combo.tensor([1.;2.])
            let t3s9Correct  = combo.tensor([1.;3.])
            let t3s10Correct = combo.tensor([[1.;2.];[3.;4.]])
            let t3s11Correct = combo.tensor([1.;5.])
            let t3s12Correct = combo.tensor([[1.;2.];[5.;6.]])
            let t3s13Correct = combo.tensor([[1.;3.];[5.;7.]])
            let t3s14Correct = combo.tensor([[[1.;2.];[3.;4.]];[[5.;6.];[7.;8.]]])

            let t4 = combo.tensor([[[[1.]]; 
                                     [[2.]]; 
                                     [[3.]]]; 
                                    [[[4.]]; 
                                     [[5.]]; 
                                     [[6.]]]])
            let t4s1 = t4.[0]
            let t4s2 = t4.[0,*,*,*]
            let t4s1Correct = combo.tensor([[[1]];
                                             [[2]];
                                             [[3]]])
            let t4s2Correct = t4s1Correct

            Assert.AreEqual(t1s1Correct, t1s1)
            Assert.AreEqual(t1s2Correct, t1s2)

            Assert.AreEqual(t2s1Correct, t2s1)
            Assert.AreEqual(t2s2Correct, t2s2)
            Assert.AreEqual(t2s3Correct, t2s3)
            Assert.AreEqual(t2s4Correct, t2s4)
            Assert.AreEqual(t2s5Correct, t2s5)
            Assert.AreEqual(t2s6Correct, t2s6)

            Assert.AreEqual(t2bs1Correct, t2bs1)
            Assert.AreEqual(t2bs2Correct, t2bs2)

            Assert.AreEqual(t3s1Correct, t3s1)
            Assert.AreEqual(t3s2Correct, t3s2)
            Assert.AreEqual(t3s3Correct, t3s3)
            Assert.AreEqual(t3s4Correct, t3s4)
            Assert.AreEqual(t3s5Correct, t3s5)
            Assert.AreEqual(t3s6Correct, t3s6)
            Assert.AreEqual(t3s7Correct, t3s7)
            Assert.AreEqual(t3s8Correct, t3s8)
            Assert.AreEqual(t3s9Correct, t3s9)
            Assert.AreEqual(t3s10Correct, t3s10)
            Assert.AreEqual(t3s11Correct, t3s11)
            Assert.AreEqual(t3s12Correct, t3s12)
            Assert.AreEqual(t3s13Correct, t3s13)
            Assert.AreEqual(t3s14Correct, t3s14)

            Assert.AreEqual(t4s1Correct, t4s1)
            Assert.AreEqual(t4s2Correct, t4s2)

            Assert.AreEqual(t1s1.dtype, combo.dtype)
            Assert.AreEqual(t1s2.dtype, combo.dtype)

            Assert.AreEqual(t2s1.dtype, combo.dtype)
            Assert.AreEqual(t2s2.dtype, combo.dtype)
            Assert.AreEqual(t2s3.dtype, combo.dtype)
            Assert.AreEqual(t2s4.dtype, combo.dtype)
            Assert.AreEqual(t2s5.dtype, combo.dtype)
            Assert.AreEqual(t2s6.dtype, combo.dtype)

            Assert.AreEqual(t2bs1.dtype, combo.dtype)
            Assert.AreEqual(t2bs2.dtype, combo.dtype)

            Assert.AreEqual(t3s1.dtype, combo.dtype)
            Assert.AreEqual(t3s2.dtype, combo.dtype)
            Assert.AreEqual(t3s3.dtype, combo.dtype)
            Assert.AreEqual(t3s4.dtype, combo.dtype)
            Assert.AreEqual(t3s5.dtype, combo.dtype)
            Assert.AreEqual(t3s6.dtype, combo.dtype)
            Assert.AreEqual(t3s7.dtype, combo.dtype)
            Assert.AreEqual(t3s8.dtype, combo.dtype)
            Assert.AreEqual(t3s9.dtype, combo.dtype)
            Assert.AreEqual(t3s10.dtype, combo.dtype)
            Assert.AreEqual(t3s11.dtype, combo.dtype)
            Assert.AreEqual(t3s12.dtype, combo.dtype)
            Assert.AreEqual(t3s13.dtype, combo.dtype)
            Assert.AreEqual(t3s14.dtype, combo.dtype)

    [<Test>]
    member _.TestTensorAddTTSlice () =
        for combo in Combos.FloatingPoint do 
            let t1 = combo.tensor([[-0.2754;  0.0172;  0.7105];
                [-0.1890;  1.7664;  0.5377];
                [-0.5313; -2.2530; -0.6235];
                [ 0.6776;  1.5844; -0.5686]])
            let t2 = combo.tensor([[-111.8892;   -7.0328];
                [  18.7557;  -86.2308]])
            let t3 = t1.addSlice([0;1], t2)
            let t3Correct = combo.tensor([[  -0.2754; -111.8720;   -6.3222];
                [  -0.1890;   20.5221;  -85.6932];
                [  -0.5313;   -2.2530;   -0.6235];
                [   0.6776;    1.5844;   -0.5686]])

            Assert.True(t3.allclose(t3Correct, 0.01))
            Assert.AreEqual(t3.dtype, combo.dtype)

    [<Test>]
    member _.TestTensorPad () =
        for combo in Combos.All do
            let t1 = combo.tensor([1.,2.,3.])
            let t1p0 = dsharp.pad(t1, [0])
            let t1p0Correct = combo.tensor([1.,2.,3.])
            let t1p1 = dsharp.pad(t1, [1])
            let t1p1Correct = combo.tensor([0.,1.,2.,3.,0.])
            let t1p2 = dsharp.pad(t1, [2])
            let t1p2Correct = combo.tensor([0.,0.,1.,2.,3.,0.,0.])
            let t2 = combo.tensor([[1.,2.,3.], [4.,5.,6.]])
            let t2p00 = dsharp.pad(t2, [0;0])
            let t2p00Correct = combo.tensor([[1.,2.,3.], [4.,5.,6.]])
            let t2p12 = dsharp.pad(t2, [1;2])
            let t2p12Correct = combo.tensor([[0, 0, 0, 0, 0, 0, 0],
                                              [0, 0, 1, 2, 3, 0, 0],
                                              [0, 0, 4, 5, 6, 0, 0],
                                              [0, 0, 0, 0, 0, 0, 0]])
            let t2p22 = dsharp.pad(t2, [2;2])
            let t2p22Correct = combo.tensor([[0, 0, 0, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0, 0, 0],
                                                [0, 0, 1, 2, 3, 0, 0],
                                                [0, 0, 4, 5, 6, 0, 0],
                                                [0, 0, 0, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0, 0, 0]])
            Assert.AreEqual(t1p0Correct, t1p0)
            Assert.AreEqual(t1p1Correct, t1p1)
            Assert.AreEqual(t1p2Correct, t1p2)
            Assert.AreEqual(t2p00Correct, t2p00)
            Assert.AreEqual(t2p12Correct, t2p12)
            Assert.AreEqual(t2p22Correct, t2p22)


    [<Test>]
    member _.TestTensorExpandT () =
        for combo in Combos.All do 
            let t1 = combo.tensor(1.0)
            let t1Expand = t1.expand([2;3])
            let t1ExpandCorrect = combo.tensor([[1.;1.;1.];[1.;1.;1.]])
            Assert.AreEqual(t1ExpandCorrect, t1Expand)

            let t2 = combo.tensor([1.0])
            let t2Expand = t2.expand([2;3])
            let t2ExpandCorrect = combo.tensor([[1.;1.;1.];[1.;1.;1.]])

            Assert.AreEqual(t2ExpandCorrect, t2Expand)

            let t3 = combo.tensor([1.; 2.]) // 2
            let t3Expand = t3.expand([3;2]) // 3x2
            let t3ExpandCorrect = combo.tensor([[1.;2.];[1.;2.];[1.;2.]]) // 3x2

            Assert.AreEqual(t3ExpandCorrect, t3Expand)

            let t4 = combo.tensor([[1.]; [2.]]) // 2x1
            let t4Expand = t4.expand([2;2]) // 2x2
            let t4ExpandCorrect = combo.tensor([[1.;1.];[2.;2.]])

            Assert.AreEqual(t4ExpandCorrect, t4Expand)

            let t5 = combo.tensor([[1.]; [2.]]) // 2x1
            let t5Expand = t5.expand([2;2;2]) // 2x2x2
            let t5ExpandCorrect = combo.tensor([[[1.;1.];[2.;2.]];[[1.;1.];[2.;2.]]])

            Assert.AreEqual(t5ExpandCorrect, t5Expand)

    [<Test>]
    member _.TestTensorSqueezeT () =
        for combo in Combos.All do 
            let t1 = combo.tensor([[[1.; 2.]]; [[3.;4.]]])
            let t1Squeeze = t1.squeeze()
            let t1SqueezeCorrect = combo.tensor([[1.;2.];[3.;4.]])

            Assert.True(t1Squeeze.allclose(t1SqueezeCorrect, 0.01))
            Assert.AreEqual(t1Squeeze.dtype, combo.dtype)

    [<Test>]
    member _.TestTensorUnsqueezeT () =
        for combo in Combos.All do 
            let t1 = combo.tensor([[1.;2.];[3.;4.]])
            let t1Unsqueeze = t1.unsqueeze(1)
            let t1UnsqueezeCorrect = combo.tensor([[[1.; 2.]]; [[3.;4.]]])

            Assert.True(t1Unsqueeze.allclose(t1UnsqueezeCorrect, 0.01))
            Assert.AreEqual(t1Unsqueeze.dtype, combo.dtype)

    [<Test>]
    member _.TestTensorFlipT () =
        for combo in Combos.All do 
            let t1 = combo.tensor([[1.;2.];[3.;4.]])
            let t2 = t1.flip([|0|])
            let t2Correct = combo.tensor([[3.;4.]; [1.;2.]])
            let t3 = t1.flip([|1|])
            let t3Correct = combo.tensor([[2.;1.]; [4.;3.]])
            let t4 = t1.flip([|0; 1|])
            let t4Correct = combo.tensor([[4.;3.]; [2.;1.]])
            let t5 = t1.flip([|0; 1|]).flip([|0; 1|])
            let t5Correct = combo.tensor([[1.;2.]; [3.;4.]])

            Assert.AreEqual(t2Correct, t2)
            Assert.AreEqual(t3Correct, t3)
            Assert.AreEqual(t4Correct, t4)
            Assert.AreEqual(t5Correct, t5)

    [<Test>]
    member _.TestTensorDilateT () =
        for combo in Combos.FloatingPoint do 
            let tin1 = combo.tensor([1.;2.;3.])
            let t1 = tin1.dilate([|2|])
            let t1Correct = combo.tensor([1.;0.;2.;0.;3.])

            Assert.AreEqual(t1Correct, t1)

            let tin2 = combo.tensor([[1.;2.]; [3.;4.]])
            let t2 = tin2.dilate([|1; 2|])
            let t2Correct = combo.tensor([[1.;0.;2.];[3.;0.;4.]])

            Assert.AreEqual(t2Correct, t2)
            Assert.AreEqual(combo.dtype, t2.dtype)

            let t3 = tin2.dilate([|2; 2|])
            let t3Correct = combo.tensor([[1.;0.;2.];[0.;0.;0.];[3.;0.;4.]])

            Assert.AreEqual(t3Correct, t3)
            Assert.AreEqual(combo.dtype, t3.dtype)

            let tin5 = combo.tensor([1.;2.;3.;4.])
            let t5 = tin5.dilate([|3|])
            let t5Correct = combo.tensor([|1.;0.;0.;2.;0.;0.;3.;0.;0.;4.|])

            Assert.AreEqual(t5Correct, t5)
            Assert.AreEqual(combo.dtype, t5.dtype)

            // Dilate 3D 1; 1; 2
            let tin6 = combo.tensor([[[1.;2.]; [3.;4.]];[[5.;6.]; [7.;8.]]])
            let t6 = tin6.dilate([|1; 1; 2|])
            let t6Correct = combo.tensor([[[1.;0.;2.];[3.;0.;4.]]; [[5.;0.;6.];[7.;0.;8.]]])

            Assert.AreEqual(t6Correct, t6)
            Assert.AreEqual(combo.dtype, t6.dtype)

            // Dilate 4D 1; 1; 1; 2
            let tin7 = combo.tensor([[[[1.;2.]; [3.;4.]];[[5.;6.]; [7.;8.]]];[[[1.;2.]; [3.;4.]];[[5.;6.]; [7.;8.]]]])
            let t7 = tin7.dilate([|1; 1; 1; 2|])
            let t7Correct = combo.tensor([[[[1.;0.;2.];[3.;0.;4.]]; [[5.;0.;6.];[7.;0.;8.]]]; [[[1.;0.;2.];[3.;0.;4.]]; [[5.;0.;6.];[7.;0.;8.]]]])

            Assert.AreEqual(t7Correct, t7)
            Assert.AreEqual(combo.dtype, t7.dtype)

            // 3D and 4D dilations not working correctly in LibTorch except when d0, d1 dilations are 1
            if combo.backend <> Backend.Torch then
                let tin8 = combo.tensor([[[1.;2.]; [3.;4.]];[[5.;6.]; [7.;8.]]])
                let t8 = tin8.dilate([|2; 1; 2|])
                let t8Correct = combo.tensor([[[1.;0.;2.];[3.;0.;4.]]; [[0.;0.;0.];[0.;0.;0.]]; [[5.;0.;6.];[7.;0.;8.]]])

                Assert.AreEqual(t8Correct, t8)
                Assert.AreEqual(combo.dtype, t8.dtype)

                // Dilate 4D, 2; 1; 1; 2
                let tin9 = combo.tensor([[[[1.;2.]; [3.;4.]];[[5.;6.]; [7.;8.]]];[[[1.;2.]; [3.;4.]];[[5.;6.]; [7.;8.]]]])
                let t9 = tin9.dilate([|2; 1; 1; 2|])
                let t9Correct = combo.tensor([[[[1.;0.;2.];[3.;0.;4.]]; [[5.;0.;6.];[7.;0.;8.]]]; 
                                              [[[0.;0.;0.];[0.;0.;0.]]; [[0.;0.;0.];[0.;0.;0.]]]; 
                                              [[[1.;0.;2.];[3.;0.;4.]]; [[5.;0.;6.];[7.;0.;8.]]]])

                Assert.AreEqual(t9Correct, t9)
                Assert.AreEqual(combo.dtype, t9.dtype)

    [<Test>]
    member _.TestTensorUndilateT () =
        for combo in Combos.All do 
            let t1 = combo.tensor([[1.;0.;2.];[3.;0.;4.]])
            let t2 = t1.undilate([|1; 2|])
            let t2Correct = combo.tensor([[1.;2.]; [3.;4.]])
            let t3 = combo.tensor([[1.;0.;2.];[0.;0.;0.];[3.;0.;4.]])
            let t4 = t3.undilate([|2; 2|])
            let t4Correct = combo.tensor([[1.;2.]; [3.;4.]])
            let t5 = combo.tensor([|1.;0.;0.;2.;0.;0.;3.;0.;0.;4.|])
            let t6 = t5.undilate([|3|])
            let t6Correct = combo.tensor([1.;2.;3.;4.])

            Assert.AreEqual(t2Correct, t2)
            Assert.AreEqual(t4Correct, t4)
            Assert.AreEqual(t6Correct, t6)
            Assert.AreEqual(combo.dtype, t2.dtype)
            Assert.AreEqual(combo.dtype, t4.dtype)
            Assert.AreEqual(combo.dtype, t6.dtype)

    [<Test>]
    member _.TestTensorView () =
        for combo in Combos.All do 
            let t = combo.randint(0, 2, [10;10])
            let t1 = t.view(-1)
            let t1Shape = t1.shape
            let t1ShapeCorrect = [|100|]
            let t2Shape = t.view([-1;50]).shape
            let t2ShapeCorrect = [|2;50|]
            let t3Shape = t.view([2;-1;50]).shape
            let t3ShapeCorrect = [|2;1;50|]
            let t4Shape = t.view([2;-1;10]).shape
            let t4ShapeCorrect = [|2;5;10|]
        
            Assert.AreEqual(t1ShapeCorrect, t1Shape)
            Assert.AreEqual(t2ShapeCorrect, t2Shape)
            Assert.AreEqual(t3ShapeCorrect, t3Shape)
            Assert.AreEqual(t4ShapeCorrect, t4Shape)
            Assert.AreEqual(t1.dtype, combo.dtype)

    [<Test>]
    member _.TestTensorFlatten () =
        for combo in Combos.All do 
            let t1 = combo.randint(0, 2, [5;5;5;5])
            let t1f1shape = dsharp.flatten(t1).shape
            let t1f1shapeCorrect = [|625|]
            let t1f2shape = dsharp.flatten(t1, startDim=1).shape
            let t1f2shapeCorrect = [|5; 125|]
            let t1f3shape = dsharp.flatten(t1, startDim=1, endDim=2).shape
            let t1f3shapeCorrect = [|5; 25; 5|]

            let t2 = combo.randint(0, 2, 5)
            let t2fshape = dsharp.flatten(t2).shape
            let t2fshapeCorrect = [|5|]

            let t3 = combo.tensor(2.5)
            let t3fshape = dsharp.flatten(t3).shape
            let t3fshapeCorrect = [||]

            Assert.AreEqual(t1f1shapeCorrect, t1f1shape)
            Assert.AreEqual(t1f2shapeCorrect, t1f2shape)
            Assert.AreEqual(t1f3shapeCorrect, t1f3shape)
            Assert.AreEqual(t2fshapeCorrect, t2fshape)
            Assert.AreEqual(t3fshapeCorrect, t3fshape)

    [<Test>]
    member _.TestTensorGather () =
        for combo in Combos.All do 
            let t1 = combo.tensor([1,2,3,4,5])
            let t1g = dsharp.gather(t1, 0, combo.tensor([0,2,3], dtype=DType.Int32))
            let t1gCorrect = combo.tensor([1, 3, 4])

            let t2 = combo.tensor([[1,2],[3,4]])
            let t2g0 = dsharp.gather(t2, 0, combo.tensor([[0,1],[1,0]], dtype=DType.Int32))
            let t2g0Correct = combo.tensor([[1, 4],
                                             [3, 2]])
            let t2g1 = dsharp.gather(t2, 1, combo.tensor([[0,0,1],[1,0,0]], dtype=DType.Int32))
            let t2g1Correct = combo.tensor([[1, 1, 2],
                                             [4, 3, 3]])

            Assert.AreEqual(t1gCorrect, t1g)
            Assert.AreEqual(combo.dtype, t1g.dtype)

            Assert.AreEqual(t2g0Correct, t2g0)
            Assert.AreEqual(combo.dtype, t2g0.dtype)

            Assert.AreEqual(t2g1Correct, t2g1)
            Assert.AreEqual(combo.dtype, t2g1.dtype)

    [<Test>]
    member _.TestTensorMax () =
        for combo in Combos.All do 
            let t1 = combo.tensor([4.;1.;20.;3.])
            let t1Max = t1.max()
            let t1MaxCorrect = combo.tensor(20.)

            let t2 = combo.tensor([[1.;4.];[2.;3.]])
            let t2Max = t2.max()
            let t2MaxCorrect = combo.tensor(4.)

            let t3 = combo.tensor([[[ 7.6884; 65.9125;  4.0114];
                                 [46.7944; 61.5331; 40.1627];
                                 [48.3240;  4.9910; 50.1571]];

                                [[13.4777; 65.7656; 36.8161];
                                 [47.8268; 42.2229;  5.6115];
                                 [43.4779; 77.8675; 95.7660]];

                                [[59.8422; 47.1146; 36.7614];
                                 [71.6328; 18.5912; 27.7328];
                                 [49.9120; 60.3023; 53.0838]]])

            let t3Max = t3.max()
            let t3MaxCorrect = combo.tensor(95.7660)
        
            let t4 = combo.tensor([[[[8.8978; 8.0936];
                                  [4.8087; 1.0921];
                                  [8.5664; 3.7814]];

                                 [[2.3581; 3.7361];
                                  [1.0436; 6.0353];
                                  [7.7843; 8.7153]];

                                 [[3.9188; 6.7906];
                                  [9.1242; 4.8711];
                                  [1.7870; 9.7456]];
                                 [[5.0444; 0.5447];
                                  [6.2945; 5.9047];
                                  [8.0867; 3.1606]]]])

            let t4Max = t4.max()
            let t4MaxCorrect = combo.tensor(9.7456)

            Assert.AreEqual(t1MaxCorrect, t1Max)
            Assert.AreEqual(t2MaxCorrect, t2Max)
            Assert.AreEqual(t3MaxCorrect, t3Max)
            Assert.AreEqual(t4MaxCorrect, t4Max)
            Assert.AreEqual(t1Max.dtype, combo.dtype)
            Assert.AreEqual(t2Max.dtype, combo.dtype)
            Assert.AreEqual(t3Max.dtype, combo.dtype)
            Assert.AreEqual(t4Max.dtype, combo.dtype)


    [<Test>]
    member _.TestTensorMin () =
        for combo in Combos.SignedIntegralAndFloatingPoint do 
            let t1 = combo.tensor([4.;1.;20.;3.])
            let t1Min = t1.min()
            let t1MinCorrect = combo.tensor(1.)

            let t2 = combo.tensor([[1.;4.];[2.;3.]])
            let t2Min = t2.min()
            let t2MinCorrect = combo.tensor(1.)

            let t3 = combo.tensor([[[ 7.6884; 65.9125;  4.0114];
                 [46.7944; 61.5331; 40.1627];
                 [48.3240;  4.9910; 50.1571]];

                [[13.4777; 65.7656; 36.8161];
                 [47.8268; 42.2229;  5.6115];
                 [43.4779; 77.8675; 95.7660]];

                [[59.8422; 47.1146; 36.7614];
                 [71.6328; 18.5912; 27.7328];
                 [49.9120; 60.3023; 53.0838]]])
            let t3Min = t3.min()
            let t3MinCorrect = combo.tensor(4.0114)
       
            let t4 = combo.tensor([[[[8.8978; 8.0936];
                  [4.8087; 1.0921];
                  [8.5664; 3.7814]];

                 [[2.3581; 3.7361];
                  [1.0436; 6.0353];
                  [7.7843; 8.7153]];

                 [[3.9188; 6.7906];
                  [9.1242; 4.8711];
                  [1.7870; 9.7456]];

                 [[5.7825; 8.0450];
                  [2.7801; 1.0877];
                  [3.4042; 5.1911]]];

                [[[0.5370; 7.1115];
                  [5.4971; 2.3567];
                  [0.9318; 8.6992]];

                 [[3.3796; 8.7833];
                  [5.8722; 5.9881];
                  [0.7646; 7.3685]];

                 [[7.5344; 9.6162];
                  [2.6404; 4.3938];
                  [3.1335; 7.6783]];

                 [[5.0444; 0.5447];
                  [6.2945; 5.9047];
                  [8.0867; 3.1606]]]])
            let t4Min = t4.min()
            let t4MinCorrect = combo.tensor(0.5370)

            Assert.AreEqual(t1MinCorrect, t1Min)
            Assert.AreEqual(t2MinCorrect, t2Min)
            Assert.AreEqual(t3MinCorrect, t3Min)
            Assert.AreEqual(t4MinCorrect, t4Min)
            Assert.AreEqual(t1Min.dtype, combo.dtype)
            Assert.AreEqual(t2Min.dtype, combo.dtype)
            Assert.AreEqual(t3Min.dtype, combo.dtype)
            Assert.AreEqual(t4Min.dtype, combo.dtype)

    [<Test>]
    member _.TestTensorMaxBinary () =
        for combo in Combos.FloatingPoint do 
            let t1 = combo.tensor([[-4.9385; 12.6206; 10.1783];
                [-2.9624; 17.6992;  2.2506];
                [-2.3536;  8.0772; 13.5639]])
            let t2 = combo.tensor([[  0.7027;  22.3251; -11.4533];
                [  3.6887;   4.3355;   3.3767];
                [  0.1203;  -5.4088;   1.5658]])
            let t3 = t1.max(t2)
            let t3Correct = combo.tensor([[ 0.7027; 22.3251; 10.1783];
                [ 3.6887; 17.6992;  3.3767];
                [ 0.1203;  8.0772; 13.5639]])

            Assert.True(t3.allclose(t3Correct, 0.01))
            Assert.AreEqual(t3.dtype, combo.dtype)

    [<Test>]
    member _.TestTensorMinBinary () =
        for combo in Combos.FloatingPoint do 
            let t1 = combo.tensor([[-4.9385; 12.6206; 10.1783];
                [-2.9624; 17.6992;  2.2506];
                [-2.3536;  8.0772; 13.5639]])
            let t2 = combo.tensor([[  0.7027;  22.3251; -11.4533];
                [  3.6887;   4.3355;   3.3767];
                [  0.1203;  -5.4088;   1.5658]])
            let t3 = t1.min(t2)
            let t3Correct = combo.tensor([[ -4.9385;  12.6206; -11.4533];
                [ -2.9624;   4.3355;   2.2506];
                [ -2.3536;  -5.4088;   1.5658]])

            Assert.True(t3.allclose(t3Correct, 0.01))
            Assert.AreEqual(t3.dtype, combo.dtype)

    [<Test>]
    member _.TestTensorSoftmax () =
        for combo in Combos.FloatingPoint do 
            let t1 = combo.tensor([2.7291; 0.0607; 0.8290])
            let t1Softmax0 = t1.softmax(0)
            let t1Softmax0Correct = combo.tensor([0.8204; 0.0569; 0.1227])

            let t2 = combo.tensor([[1.3335; 1.6616; 2.4874; 6.1722];
                [3.3478; 9.3019; 1.0844; 8.9874];
                [8.6300; 1.8842; 9.1387; 9.1321]])
            let t2Softmax0 = t2.softmax(0)
            let t2Softmax0Correct = combo.tensor([[6.7403e-04; 4.8014e-04; 1.2904e-03; 2.7033e-02];
                [5.0519e-03; 9.9892e-01; 3.1723e-04; 4.5134e-01];
                [9.9427e-01; 5.9987e-04; 9.9839e-01; 5.2163e-01]])
            let t2Softmax1 = t2.softmax(1)
            let t2Softmax1Correct = combo.tensor([[7.5836e-03; 1.0528e-02; 2.4044e-02; 9.5784e-01];
                [1.4974e-03; 5.7703e-01; 1.5573e-04; 4.2131e-01];
                [2.3167e-01; 2.7240e-04; 3.8528e-01; 3.8277e-01]])

            let t3 = combo.tensor([[[3.0897; 2.0902];
                 [2.4055; 1.2437];
                 [2.1253; 8.7802];
                 [4.3856; 3.4456]];

                [[8.6233; 6.9789];
                 [4.9583; 9.9497];
                 [2.6964; 1.6048];
                 [2.1182; 2.1071]];

                [[8.1097; 6.9804];
                 [8.1223; 6.3030];
                 [0.1873; 8.7840];
                 [9.3609; 0.6493]]])
             
            let t3Softmax0 = t3.softmax(0)
            let t3Softmax0Correct = combo.tensor([[[2.4662e-03; 3.7486e-03];
                 [3.1467e-03; 1.6136e-04];
                 [3.4316e-01; 4.9885e-01];
                 [6.8542e-03; 7.5571e-01]];

                [[6.2411e-01; 4.9776e-01];
                 [4.0415e-02; 9.7443e-01];
                 [6.0743e-01; 3.8170e-04];
                 [7.0995e-04; 1.9817e-01]];

                [[3.7342e-01; 4.9849e-01];
                 [9.5644e-01; 2.5410e-02];
                 [4.9412e-02; 5.0077e-01];
                 [9.9244e-01; 4.6122e-02]]])
            let t3Softmax1 = t3.softmax(1)
            let t3Softmax1Correct = combo.tensor([[[1.8050e-01; 1.2351e-03];
                 [9.1058e-02; 5.2978e-04];
                 [6.8813e-02; 9.9344e-01];
                 [6.5963e-01; 4.7904e-03]];

                [[9.7109e-01; 4.8732e-02];
                 [2.4864e-02; 9.5067e-01];
                 [2.5896e-03; 2.2587e-04];
                 [1.4526e-03; 3.7327e-04]];

                [[1.8156e-01; 1.3190e-01];
                 [1.8387e-01; 6.6997e-02];
                 [6.5824e-05; 8.0087e-01];
                 [6.3451e-01; 2.3479e-04]]])
            let t3Softmax2 = t3.softmax(2)
            let t3Softmax2Correct = combo.tensor([[[7.3096e-01; 2.6904e-01];
                 [7.6165e-01; 2.3835e-01];
                 [1.2861e-03; 9.9871e-01];
                 [7.1910e-01; 2.8090e-01]];

                [[8.3814e-01; 1.6186e-01];
                 [6.7502e-03; 9.9325e-01];
                 [7.4868e-01; 2.5132e-01];
                 [5.0278e-01; 4.9722e-01]];

                [[7.5571e-01; 2.4429e-01];
                 [8.6049e-01; 1.3951e-01];
                 [1.8468e-04; 9.9982e-01];
                 [9.9984e-01; 1.6463e-04]]])

            Assert.True(t1Softmax0.allclose(t1Softmax0Correct, 0.001))
            Assert.True(t2Softmax0.allclose(t2Softmax0Correct, 0.001))
            Assert.True(t2Softmax1.allclose(t2Softmax1Correct, 0.001))
            Assert.True(t3Softmax0.allclose(t3Softmax0Correct, 0.001))
            Assert.True(t3Softmax1.allclose(t3Softmax1Correct, 0.001))
            Assert.True(t3Softmax2.allclose(t3Softmax2Correct, 0.001))
            Assert.AreEqual(t1Softmax0.dtype, combo.dtype)
            Assert.AreEqual(t2Softmax0.dtype, combo.dtype)
            Assert.AreEqual(t2Softmax1.dtype, combo.dtype)
            Assert.AreEqual(t3Softmax0.dtype, combo.dtype)
            Assert.AreEqual(t3Softmax1.dtype, combo.dtype)
            Assert.AreEqual(t3Softmax2.dtype, combo.dtype)


    [<Test>]
    member _.TestTensorLogsoftmax () =
        for combo in Combos.FloatingPoint do 
            let t1 = combo.tensor([2.7291, 0.0607, 0.8290])
            let t1Logsoftmax0 = t1.logsoftmax(0)
            let t1Logsoftmax0Correct = combo.tensor([-0.1980, -2.8664, -2.0981])

            let t2 = combo.tensor([[1.3335, 1.6616, 2.4874, 6.1722],
                                    [3.3478, 9.3019, 1.0844, 8.9874],
                                    [8.6300, 1.8842, 9.1387, 9.1321]])
            let t2Logsoftmax0 = t2.logsoftmax(0)
            let t2Logsoftmax0Correct = combo.tensor([[-7.3022e+00, -7.6414e+00, -6.6529e+00, -3.6107e+00],
                                                        [-5.2879e+00, -1.0806e-03, -8.0559e+00, -7.9552e-01],
                                                        [-5.7426e-03, -7.4188e+00, -1.6088e-03, -6.5082e-01]])
            let t2Logsoftmax1 = t2.logsoftmax(1)
            let t2Logsoftmax1Correct = combo.tensor([[-4.8818, -4.5537, -3.7279, -0.0431],
                                                        [-6.5040, -0.5499, -8.7674, -0.8644],
                                                        [-1.4624, -8.2082, -0.9537, -0.9603]])

            let t3 = combo.tensor([[[3.0897, 2.0902],
                                     [2.4055, 1.2437],
                                     [2.1253, 8.7802],
                                     [4.3856, 3.4456]],

                                    [[8.6233, 6.9789],
                                     [4.9583, 9.9497],
                                     [2.6964, 1.6048],
                                     [2.1182, 2.1071]],

                                    [[8.1097, 6.9804],
                                     [8.1223, 6.3030],
                                     [0.1873, 8.7840],
                                     [9.3609, 0.6493]]])
             
            let t3Logsoftmax0 = t3.logsoftmax(0)
            let t3Logsoftmax0Correct = combo.tensor([[[-6.0050e+00, -5.5864e+00],
                                                         [-5.7613e+00, -8.7319e+00],
                                                         [-1.0696e+00, -6.9543e-01],
                                                         [-4.9829e+00, -2.8011e-01]],

                                                        [[-4.7143e-01, -6.9765e-01],
                                                         [-3.2085e+00, -2.5904e-02],
                                                         [-4.9850e-01, -7.8708e+00],
                                                         [-7.2503e+00, -1.6186e+00]],

                                                        [[-9.8503e-01, -6.9615e-01],
                                                         [-4.4540e-02, -3.6726e+00],
                                                         [-3.0076e+00, -6.9163e-01],
                                                         [-7.5929e-03, -3.0764e+00]]])
            let t3Logsoftmax1 = t3.logsoftmax(1)
            let t3Logsoftmax1Correct = combo.tensor([[[-1.7120e+00, -6.6966e+00],
                                                         [-2.3962e+00, -7.5431e+00],
                                                         [-2.6764e+00, -6.5767e-03],
                                                         [-4.1609e-01, -5.3412e+00]],

                                                        [[-2.9332e-02, -3.0214e+00],
                                                         [-3.6943e+00, -5.0591e-02],
                                                         [-5.9562e+00, -8.3955e+00],
                                                         [-6.5344e+00, -7.8932e+00]],

                                                        [[-1.7061e+00, -2.0257e+00],
                                                         [-1.6935e+00, -2.7031e+00],
                                                         [-9.6285e+00, -2.2207e-01],
                                                         [-4.5492e-01, -8.3568e+00]]])
            let t3Logsoftmax2 = t3.logsoftmax(2)
            let t3Logsoftmax2Correct = combo.tensor([[[-3.1340e-01, -1.3129e+00],
                                                         [-2.7226e-01, -1.4341e+00],
                                                         [-6.6562e+00, -1.2869e-03],
                                                         [-3.2976e-01, -1.2698e+00]],

                                                        [[-1.7658e-01, -1.8210e+00],
                                                         [-4.9982e+00, -6.7731e-03],
                                                         [-2.8944e-01, -1.3810e+00],
                                                         [-6.8761e-01, -6.9871e-01]],

                                                        [[-2.8010e-01, -1.4094e+00],
                                                         [-1.5026e-01, -1.9696e+00],
                                                         [-8.5969e+00, -1.8464e-04],
                                                         [-1.6461e-04, -8.7118e+00]]])
            Assert.True(t1Logsoftmax0.allclose(t1Logsoftmax0Correct, 0.01))
            Assert.True(t2Logsoftmax0.allclose(t2Logsoftmax0Correct, 0.01))
            Assert.True(t2Logsoftmax1.allclose(t2Logsoftmax1Correct, 0.01))
            Assert.True(t3Logsoftmax0.allclose(t3Logsoftmax0Correct, 0.01))
            Assert.True(t3Logsoftmax1.allclose(t3Logsoftmax1Correct, 0.01))
            Assert.True(t3Logsoftmax2.allclose(t3Logsoftmax2Correct, 0.01))

    [<Test>]
    member _.TestTensorLogsumexp () =
        for combo in Combos.FloatingPoint do 
            let t1 = combo.tensor([2.7291, 0.0607, 0.8290])
            let t1Logsumexp0 = t1.logsumexp(0)
            let t1Logsumexp0Correct = combo.tensor(2.9271)
            let t1Logsumexp0keepdim = t1.logsumexp(0, keepDim=true)
            let t1Logsumexp0keepdimCorrect = combo.tensor([2.9271])

            let t2 = combo.tensor([[1.3335, 1.6616, 2.4874, 6.1722],
                                    [3.3478, 9.3019, 1.0844, 8.9874],
                                    [8.6300, 1.8842, 9.1387, 9.1321]])
            let t2Logsumexp0 = t2.logsumexp(0)
            let t2Logsumexp0Correct = combo.tensor([8.6357, 9.3030, 9.1403, 9.7829])
            let t2Logsumexp0keepdim = t2.logsumexp(0, keepDim=true)
            let t2Logsumexp0keepdimCorrect = combo.tensor([[8.6357, 9.3030, 9.1403, 9.7829]])
            let t2Logsumexp1 = t2.logsumexp(1)
            let t2Logsumexp1Correct = combo.tensor([ 6.2153,  9.8518, 10.0924])
            let t2Logsumexp1keepdim = t2.logsumexp(1, keepDim=true)
            let t2Logsumexp1keepdimCorrect = combo.tensor([[ 6.2153],
                                                            [ 9.8518],
                                                            [10.0924]])

            let t3 = combo.tensor([[[3.0897, 2.0902],
                                     [2.4055, 1.2437],
                                     [2.1253, 8.7802],
                                     [4.3856, 3.4456]],

                                    [[8.6233, 6.9789],
                                     [4.9583, 9.9497],
                                     [2.6964, 1.6048],
                                     [2.1182, 2.1071]],

                                    [[8.1097, 6.9804],
                                     [8.1223, 6.3030],
                                     [0.1873, 8.7840],
                                     [9.3609, 0.6493]]])
             
            let t3Logsumexp0 = t3.logsumexp(0)
            let t3Logsumexp0Correct = combo.tensor([[9.0947, 7.6766],
                                                        [8.1668, 9.9756],
                                                        [3.1949, 9.4756],
                                                        [9.3685, 3.7257]])
            let t3Logsumexp0keepdim = t3.logsumexp(0, keepDim=true)
            let t3Logsumexp0keepdimCorrect = combo.tensor([[[9.0947, 7.6766],
                                                             [8.1668, 9.9756],
                                                             [3.1949, 9.4756],
                                                             [9.3685, 3.7257]]])                                                    
            let t3Logsumexp1 = t3.logsumexp(1)
            let t3Logsumexp1Correct = combo.tensor([[ 4.8017,  8.7868],
                                                        [ 8.6526, 10.0003],
                                                        [ 9.8158,  9.0061]])
            let t3Logsumexp1keepdim = t3.logsumexp(1, keepDim=true)
            let t3Logsumexp1keepdimCorrect = combo.tensor([[[ 4.8017,  8.7868]],

                                                            [[ 8.6526, 10.0003]],

                                                            [[ 9.8158,  9.0061]]])
            let t3Logsumexp2 = t3.logsumexp(2)
            let t3Logsumexp2Correct = combo.tensor([[3.4031, 2.6778, 8.7815, 4.7154],
                                                        [8.7999, 9.9565, 2.9858, 2.8058],
                                                        [8.3898, 8.2726, 8.7842, 9.3611]])
            let t3Logsumexp2keepdim = t3.logsumexp(2, keepDim=true)
            let t3Logsumexp2keepdimCorrect = combo.tensor([[[3.4031],
                                                             [2.6778],
                                                             [8.7815],
                                                             [4.7154]],

                                                            [[8.7999],
                                                             [9.9565],
                                                             [2.9858],
                                                             [2.8058]],

                                                            [[8.3898],
                                                             [8.2726],
                                                             [8.7842],
                                                             [9.3611]]])

            let t4 = combo.tensor([[167.385696, -146.549866, 168.850235, -41.856903, -56.691696, -78.774994, 42.035625, 97.490936, -42.763878, -2.130855], 
                                     [-62.961613, -497.529846, 371.218231, -30.224543, 368.146393, -325.945068, -292.102631, -24.760872, 130.348282, -193.775909]])
            let t4Logsumexp1 = t4.logsumexp(dim=1)
            let t4Logsumexp1Correct = combo.tensor([169.0582, 371.2635])
            Assert.True(t1Logsumexp0.allclose(t1Logsumexp0Correct, 0.001))
            Assert.True(t2Logsumexp0.allclose(t2Logsumexp0Correct, 0.001))
            Assert.True(t2Logsumexp1.allclose(t2Logsumexp1Correct, 0.001))
            Assert.True(t3Logsumexp0.allclose(t3Logsumexp0Correct, 0.001))
            Assert.True(t3Logsumexp1.allclose(t3Logsumexp1Correct, 0.001))
            Assert.True(t3Logsumexp2.allclose(t3Logsumexp2Correct, 0.001))
            Assert.True(t1Logsumexp0keepdim.allclose(t1Logsumexp0keepdimCorrect, 0.001))
            Assert.True(t2Logsumexp0keepdim.allclose(t2Logsumexp0keepdimCorrect, 0.001))
            Assert.True(t2Logsumexp1keepdim.allclose(t2Logsumexp1keepdimCorrect, 0.001))
            Assert.True(t3Logsumexp0keepdim.allclose(t3Logsumexp0keepdimCorrect, 0.001))
            Assert.True(t3Logsumexp1keepdim.allclose(t3Logsumexp1keepdimCorrect, 0.001))
            Assert.True(t3Logsumexp2keepdim.allclose(t3Logsumexp2keepdimCorrect, 0.001))
            Assert.True(t4Logsumexp1.allclose(t4Logsumexp1Correct, 0.75))

    [<Test>]
    member _.TestTensorNllLoss () =
        for combo in Combos.FloatingPoint do 
            let t1a = combo.tensor([[0.15,0.85],[0.5,0.5],[0.8,0.2]]).log()
            let t1b = combo.tensor([0,1,1])
            let t1w = combo.tensor([-1.2,0.6])
            let l1 = dsharp.nllLoss(t1a, t1b)
            let l1Correct = combo.tensor(1.3999)
            // Note, test disabled - this is not the correct answer, even on the backend
            // it was coming out as -Infinity
            //let l2 = dsharp.nllLoss(t1a, t1b, weight=t1w)
            //let l2Correct = combo.tensor(-0.8950)
            let l3 = dsharp.nllLoss(t1a, t1b, reduction="none")
            let l3Correct = combo.tensor([1.8971, 0.6931, 1.6094])
            let l4 = dsharp.nllLoss(t1a, t1b, reduction="none", weight=t1w)
            let l4Correct = combo.tensor([-2.2765,  0.4159,  0.9657])
            let l5 = dsharp.nllLoss(t1a, t1b, reduction="sum")
            let l5Correct = combo.tensor(4.1997)
            let l6 = dsharp.nllLoss(t1a, t1b, reduction="sum", weight=t1w)
            let l6Correct = combo.tensor(-0.8950)

            let t2a = combo.tensor([[[[-1.9318, -1.9386, -0.9488, -0.8787],
                                          [-1.1891, -2.4614, -1.0514, -1.1577],
                                          [-1.1977, -1.2468, -0.8123, -1.2226],
                                          [-0.9584, -2.1857, -0.9079, -1.5362]],

                                         [[-0.5465, -0.3825, -1.2375, -0.8330],
                                          [-2.4107, -0.8157, -0.9717, -1.0601],
                                          [-0.9040, -1.3655, -1.6613, -1.0334],
                                          [-0.8829, -1.4097, -1.5420, -1.9021]],

                                         [[-1.2868, -1.7491, -1.1311, -1.8975],
                                          [-0.5013, -0.7500, -1.3016, -1.0807],
                                          [-1.2271, -0.7824, -1.0044, -1.0505],
                                          [-1.5950, -0.4410, -0.9606, -0.4533]]],


                                        [[[-1.9389, -2.4012, -1.0333, -1.4381],
                                          [-1.5336, -1.6488, -2.1201, -1.5972],
                                          [-1.2268, -1.2666, -0.7287, -1.1079],
                                          [-1.3558, -1.0362, -1.2035, -1.0245]],

                                         [[-0.5721, -0.3562, -1.0314, -0.8208],
                                          [-0.4922, -0.5392, -0.9215, -0.5276],
                                          [-1.3011, -0.6734, -0.9661, -0.5593],
                                          [-0.6594, -0.9271, -1.0346, -0.7122]],

                                         [[-1.2316, -1.5651, -1.2460, -1.1315],
                                          [-1.7548, -1.4939, -0.7297, -1.5724],
                                          [-0.8335, -1.5690, -1.9886, -2.3212],
                                          [-1.4912, -1.3883, -1.0658, -1.8940]]]])
            let t2b = combo.tensor([[[2, 0, 1, 2],
                                         [2, 0, 1, 0],
                                         [2, 1, 0, 1],
                                         [1, 2, 1, 1]],

                                        [[2, 0, 2, 0],
                                         [0, 1, 0, 2],
                                         [2, 0, 2, 1],
                                         [1, 1, 1, 2]]])
            let t2w = combo.tensor([ 1.1983, -0.2633, -0.3064])
            let l7 = dsharp.nllLoss(t2a, t2b)
            let l7Correct = combo.tensor(1.3095)
            let l8 = dsharp.nllLoss(t2a, t2b, weight=t2w)
            let l8Correct = combo.tensor(2.4610)
            let l9 = dsharp.nllLoss(t2a, t2b, reduction="none")
            let l9Correct = combo.tensor([[[1.2868, 1.9386, 1.2375, 1.8975],
                                             [0.5013, 2.4614, 0.9717, 1.1577],
                                             [1.2271, 1.3655, 0.8123, 1.0334],
                                             [0.8829, 0.4410, 1.5420, 1.9021]],

                                            [[1.2316, 2.4012, 1.2460, 1.4381],
                                             [1.5336, 0.5392, 2.1201, 1.5724],
                                             [0.8335, 1.2666, 1.9886, 0.5593],
                                             [0.6594, 0.9271, 1.0346, 1.8940]]])
            let l10 = dsharp.nllLoss(t2a, t2b, reduction="none", weight=t2w)
            let l10Correct = combo.tensor([[[-0.3943,  2.3231, -0.3258, -0.5814],
                                             [-0.1536,  2.9496, -0.2558,  1.3872],
                                             [-0.3760, -0.3595,  0.9734, -0.2721],
                                             [-0.2324, -0.1351, -0.4059, -0.5007]],

                                            [[-0.3774,  2.8775, -0.3818,  1.7233],
                                             [ 1.8378, -0.1419,  2.5406, -0.4818],
                                             [-0.2554,  1.5179, -0.6093, -0.1472],
                                             [-0.1736, -0.2440, -0.2724, -0.5804]]])
            let l11 = dsharp.nllLoss(t2a, t2b, reduction="sum")
            let l11Correct = combo.tensor(41.9042)
            let l12 = dsharp.nllLoss(t2a, t2b, reduction="sum", weight=t2w)
            let l12Correct = combo.tensor(10.4726)

            Assert.True(l1Correct.allclose(l1, 0.001))
            //Assert.True(l2Correct.allclose(l2, 0.001))
            Assert.True(l3Correct.allclose(l3, 0.001))
            Assert.True(l4Correct.allclose(l4, 0.001))
            Assert.True(l5Correct.allclose(l5, 0.001))
            Assert.True(l6Correct.allclose(l6, 0.001))
            Assert.True(l7Correct.allclose(l7, 0.001))
            Assert.True(l8Correct.allclose(l8, 0.001))
            Assert.True(l9Correct.allclose(l9, 0.001))
            Assert.True(l10Correct.allclose(l10, 0.001))
            Assert.True(l11Correct.allclose(l11, 0.001))
            Assert.True(l12Correct.allclose(l12, 0.001))

    [<Test>]
    member _.TestTensorCrossEntropyLoss () =
        for combo in Combos.FloatingPoint do 
            let t1a = combo.tensor([[-0.6596,  0.3078, -0.2525, -0.2593, -0.2354],
                                        [ 0.4708,  0.6073,  1.5621, -1.4636,  0.9769],
                                        [ 0.5078,  0.0579,  1.0054,  0.3532,  1.1819],
                                        [ 1.5425, -0.2887,  1.0716, -1.3946,  0.8806]])
            let t1b = combo.tensor([3, 1, 0, 4])
            let t1w = combo.tensor([-1.4905,  0.5929,  1.0018, -1.0858, -0.5993])
            let l1 = dsharp.crossEntropyLoss(t1a, t1b)
            let l1Correct = combo.tensor(1.7059)
            let l2 = dsharp.crossEntropyLoss(t1a, t1b, weight=t1w)
            let l2Correct = combo.tensor(1.6969)
            let l3 = dsharp.crossEntropyLoss(t1a, t1b, reduction="none")
            let l3Correct = combo.tensor([1.6983, 1.7991, 1.8085, 1.5178])
            let l4 = dsharp.crossEntropyLoss(t1a, t1b, reduction="none", weight=t1w)
            let l4Correct = combo.tensor([-1.8439,  1.0666, -2.6956, -0.9096])
            let l5 = dsharp.crossEntropyLoss(t1a, t1b, reduction="sum")
            let l5Correct = combo.tensor(6.8237)
            let l6 = dsharp.crossEntropyLoss(t1a, t1b, reduction="sum", weight=t1w)
            let l6Correct = combo.tensor(-4.3825)

            Assert.True(l1Correct.allclose(l1, 0.001))
            Assert.True(l2Correct.allclose(l2, 0.001))
            Assert.True(l3Correct.allclose(l3, 0.001))
            Assert.True(l4Correct.allclose(l4, 0.001))
            Assert.True(l5Correct.allclose(l5, 0.001))
            Assert.True(l6Correct.allclose(l6, 0.001))

    [<Test>]
    member _.TestTensorMseLoss () =
        for combo in Combos.FloatingPoint do 
            let t1a = combo.tensor([-0.2425,  0.2643,  0.7070,  1.2049,  1.6245])
            let t1b = combo.tensor([-1.0742,  1.5874,  0.6509,  0.8715,  0.0692])
            let l1 = dsharp.mseLoss(t1a, t1b)
            let l1Correct = combo.tensor(0.9951)
            let l2 = dsharp.mseLoss(t1a, t1b, reduction="none")
            let l2Correct = combo.tensor([0.6917, 1.7507, 0.0031, 0.1112, 2.4190])
            let l3 = dsharp.mseLoss(t1a, t1b, reduction="sum")
            let l3Correct = combo.tensor(4.9756)

            let t2a = combo.tensor([[ 0.6650,  0.5049, -0.7356,  0.5312, -0.6574],
                                     [ 1.0133,  0.9106,  0.1523,  0.2662,  1.1438],
                                     [ 0.3641, -1.8525, -0.0822, -1.0361,  0.2723]])
            let t2b = combo.tensor([[-1.0001, -1.4867, -0.3340, -0.2590,  0.1395],
                                     [-2.0158,  0.8281,  1.1726, -0.2359,  0.5007],
                                     [ 1.3242,  0.5215,  1.4293, -1.4235,  0.2473]])
            let l4 = dsharp.mseLoss(t2a, t2b)
            let l4Correct = combo.tensor(1.8694)
            let l5 = dsharp.mseLoss(t2a, t2b, reduction="none")
            let l5Correct = combo.tensor([[2.7726e+00, 3.9663e+00, 1.6130e-01, 6.2438e-01, 6.3511e-01],
                                            [9.1753e+00, 6.8075e-03, 1.0409e+00, 2.5207e-01, 4.1352e-01],
                                            [9.2194e-01, 5.6358e+00, 2.2848e+00, 1.5011e-01, 6.2556e-04]])
            let l6 = dsharp.mseLoss(t2a, t2b, reduction="sum")
            let l6Correct = combo.tensor(28.0416)

            Assert.True(l1Correct.allclose(l1, 0.01, 0.01))
            Assert.True(l2Correct.allclose(l2, 0.01, 0.01))
            Assert.True(l3Correct.allclose(l3, 0.01, 0.01))
            Assert.True(l4Correct.allclose(l4, 0.01, 0.01))
            Assert.True(l5Correct.allclose(l5, 0.01, 0.01))
            Assert.True(l6Correct.allclose(l6, 0.01, 0.01))

    [<Test>]
    member _.TestTensorDepth () =
        for combo in Combos.All do 
            let t0 = combo.tensor([1.;2.])
            let t0Depth = t0.depth
            let t0DepthCorrect = 0
            let t1 = combo.tensor([1.;2.]).reverseDiff()
            let t1Depth = t1.depth
            let t1DepthCorrect = 1
            let t2 = combo.tensor([1.;2.]).reverseDiff().reverseDiff()
            let t2Depth = t2.depth
            let t2DepthCorrect = 2
            let t3 = combo.tensor([1.;2.]).reverseDiff().reverseDiff().forwardDiff(combo.tensor([1.; 1.]))
            let t3Depth = t3.depth
            let t3DepthCorrect = 3

            Assert.AreEqual(t0DepthCorrect, t0Depth)
            Assert.AreEqual(t1DepthCorrect, t1Depth)
            Assert.AreEqual(t2DepthCorrect, t2Depth)
            Assert.AreEqual(t3DepthCorrect, t3Depth)

    [<Test>]
    member _.FSharpCoreOps () =
        for combo in Combos.FloatingPoint do 
            let t = combo.tensor([0.1; 0.2; 0.3])
            let add = t + t
            let addCorrect = t.add(t)
            let sub = t - t
            let subCorrect = t.sub(t)
            let mul = t * t
            let mulCorrect = t.mul(t)
            let div = t / t
            let divCorrect = t.div(t)
            let pow = t ** t
            let powCorrect = t.pow(t)
            let neg = -t
            let negCorrect = t.neg()
            // sign t not supported because FSharp.Core sign operator returns int
            let floor = floor t
            let floorCorrect = t.floor()
            let ceil = ceil t
            let ceilCorrect = t.ceil()
            let round = round t
            let roundCorrect = t.round()
            let abs = abs t
            let absCorrect = t.abs()
            let exp = exp t
            let expCorrect = t.exp()
            let log = log t
            let logCorrect = t.log()
            let log10 = log10 t
            let log10Correct = t.log10()
            let sqrt = sqrt t
            let sqrtCorrect = t.sqrt()
            let sin = sin t
            let sinCorrect = t.sin()
            let cos = cos t
            let cosCorrect = t.cos()
            let tan = tan t
            let tanCorrect = t.tan()
            let sinh = sinh t
            let sinhCorrect = t.sinh()
            let cosh = cosh t
            let coshCorrect = t.cosh()
            let tanh = tanh t
            let tanhCorrect = t.tanh()
            let asin = asin t
            let asinCorrect = t.asin()
            let acos = acos t
            let acosCorrect = t.acos()
            let atan = atan t
            let atanCorrect = t.atan()
        
            Assert.AreEqual(addCorrect, add)
            Assert.AreEqual(subCorrect, sub)
            Assert.AreEqual(mulCorrect, mul)
            Assert.AreEqual(divCorrect, div)
            Assert.AreEqual(powCorrect, pow)
            Assert.AreEqual(negCorrect, neg)
            Assert.AreEqual(floorCorrect, floor)
            Assert.AreEqual(ceilCorrect, ceil)
            Assert.AreEqual(roundCorrect, round)
            Assert.AreEqual(absCorrect, abs)
            Assert.AreEqual(expCorrect, exp)
            Assert.AreEqual(logCorrect, log)
            Assert.AreEqual(log10Correct, log10)
            Assert.AreEqual(sqrtCorrect, sqrt)
            Assert.AreEqual(sinCorrect, sin)
            Assert.AreEqual(cosCorrect, cos)
            Assert.AreEqual(tanCorrect, tan)
            Assert.AreEqual(sinhCorrect, sinh)
            Assert.AreEqual(coshCorrect, cosh)
            Assert.AreEqual(tanhCorrect, tanh)
            Assert.AreEqual(asinCorrect, asin)
            Assert.AreEqual(acosCorrect, acos)
            Assert.AreEqual(atanCorrect, atan)

            Assert.AreEqual(combo.dtype, add.dtype)
            Assert.AreEqual(combo.dtype, sub.dtype)
            Assert.AreEqual(combo.dtype, mul.dtype)
            Assert.AreEqual(combo.dtype, div.dtype)
            Assert.AreEqual(combo.dtype, pow.dtype)
            Assert.AreEqual(combo.dtype, neg.dtype)
            Assert.AreEqual(combo.dtype, floor.dtype)
            Assert.AreEqual(combo.dtype, ceil.dtype)
            Assert.AreEqual(combo.dtype, round.dtype)
            Assert.AreEqual(combo.dtype, abs.dtype)
            Assert.AreEqual(combo.dtype, exp.dtype)
            Assert.AreEqual(combo.dtype, log.dtype)
            Assert.AreEqual(combo.dtype, log10.dtype)
            Assert.AreEqual(combo.dtype, sqrt.dtype)
            Assert.AreEqual(combo.dtype, sin.dtype)
            Assert.AreEqual(combo.dtype, cos.dtype)
            Assert.AreEqual(combo.dtype, tan.dtype)
            Assert.AreEqual(combo.dtype, sinh.dtype)
            Assert.AreEqual(combo.dtype, cosh.dtype)
            Assert.AreEqual(combo.dtype, tanh.dtype)
            Assert.AreEqual(combo.dtype, asin.dtype)
            Assert.AreEqual(combo.dtype, acos.dtype)
            Assert.AreEqual(combo.dtype, atan.dtype)