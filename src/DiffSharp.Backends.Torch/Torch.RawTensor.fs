namespace rec DiffSharp.Backends.Torch

open System
open DiffSharp
open DiffSharp.Backends
open DiffSharp.Util
open TorchSharp
open TorchSharp.Tensor

type TorchShape = int64[]

[<AutoOpen>]
module internal Utils = 

    let int64s (b: int[]) = Array.map int64 b

    let toTorchType dtype =
        match dtype with 
        | DType.Bool -> ScalarType.Bool
        | DType.Int8 -> ScalarType.SByte
        | DType.Byte -> ScalarType.Byte
        | DType.Int16 -> ScalarType.Short
        | DType.Int32 -> ScalarType.Int
        | DType.Int64 -> ScalarType.Long
        | DType.Float32 -> ScalarType.Float
        | DType.Float64 -> ScalarType.Double
        | DType.Other _ -> failwith "Torch GetItem TBD other type"

    let toTorchShape (shape: int[]) : TorchShape = int64s shape

    let toTorchDevice (device: Device) =
        match device with 
        | Device.CPU -> "cpu"
        | Device.GPU -> "gpu"
        | _ -> failwith "unknown device for Torch"

    let inline combineHashes (h1 : int) (h2 : int) = ((h1 <<< 5) + h1) ^^^ h2

    type RawTensor with
        member x.TorchTensor = (x :?> TorchRawTensor).TorchTensor

/// This is the base class for all RawTensorXyz tuypes.
/// All type-independent operations are implemented directly on this class. 
type TorchRawTensor(tt: TorchTensor, shape: int[], dtype, device, ?mutability) =
    inherit RawTensor(shape, dtype, device, Backend.Torch, defaultArg mutability Immutable)

    do 
       if tt.Type <> toTorchType dtype then
           failwithf "mismatched Torch tensor type, expected %A, got %A" (toTorchType dtype) tt.Type

       if toTorchShape shape <> tt.Shape then 
           failwithf "mismatched Torch tensor shape, expected %A, got %A" (toTorchShape shape) tt.Shape

    member t.MakeLike(tt, ?shape, ?dtype, ?device) : RawTensor =
        upcast TorchRawTensor(tt, defaultArg shape t.Shape, defaultArg dtype t.DType, defaultArg device t.Device)

    member x.TorchTensor = tt

    override t.GetSlice(fullBounds:int[,]) =
        let newShape = Shape.checkCanGetSlice t.Shape fullBounds
        let mutable res = tt
        let mutable dim = 0 
        for i=0 to (fullBounds.GetLength(0) - 1) do
            let start = fullBounds.[i,0]
            let stop = fullBounds.[i,1] + 1

            let len = stop - start
            let idxs = LongTensor.Arange(int64 start, int64 stop, 1L)
            res <- res.IndexSelect(int64 dim, idxs)  // yield len // if len=1 then squeeze this dimension
            if fullBounds.[i, 2] = 1 && len = 1 then 
                res <- res.Squeeze(int64 dim)  // yield len // if len=1 then squeeze this dimension
            else
                dim <- dim + 1
        t.MakeLike(tt=res, shape=newShape)

    override t.Clone() =
        t.MakeLike(tt.Clone())

   // TODO: check if torch has a C++ hashing routine
    override x.ComputeHash() = 
        let mutable res = hash shape
        let n = shapeLength shape
        match dtype with 
        | DType.Int8 ->
            let data = tt.Data<sbyte>()
            for i in 0 .. n-1 do
                 res <- combineHashes res (int32 data.[i])
        | DType.Byte ->
            let data = tt.Data<byte>()
            for i in 0 .. n-1 do
                 res <- combineHashes res (int32 data.[i])
        | DType.Bool ->
            let data = tt.Data<byte>()
            for i in 0 .. n-1 do
                 res <- combineHashes res (int32 data.[i])
        | DType.Int16 ->
            let data = tt.Data<int16>()
            for i in 0 .. n-1 do
                 res <- combineHashes res (int32 data.[i] )
        | DType.Int32 ->
            let data = tt.Data<int32>()
            for i in 0 .. n-1 do
                 res <- combineHashes res (int32 data.[i])
        | DType.Int64 -> 
            let data = tt.Data<int64>()
            for i in 0 .. n-1 do
                 res <- combineHashes res (int32 data.[i])
        | DType.Float32 ->
            let data = tt.Data<single>()
            for i in 0 .. n-1 do
                 res <- combineHashes res (hash data.[i])
        | DType.Float64 ->
            let data = tt.Data<double>()
            for i in 0 .. n-1 do
                 res <- combineHashes res (hash data.[i])
        | DType.Other _ -> failwith "Other types not supported by torch"
        res
    
    override t.Expand(newShape) =
        t.MakeLike(tt.Expand(toTorchShape newShape), shape=newShape)

    override t.GetItem(indexes) = 
        let item = 
            match indexes with 
            | [| |] -> tt
            | [| i0 |] -> tt.[int64 i0]
            | [| i0; i1 |] -> tt.[int64 i0, int64 i1]
            | [| i0; i1; i2 |] -> tt.[int64 i0, int64 i1, int64 i2]
            | [| i0; i1; i2; i3 |] -> tt.[int64 i0, int64 i1, int64 i2, int64 i3]
            | _ -> failwith "dim > 4"
        let obj = 
            match dtype with 
            | DType.Bool -> box (item.DataItem<bool>())
            | DType.Byte -> box (item.DataItem<byte>())
            | DType.Int8 -> box (item.DataItem<int8>())
            | DType.Int16 -> box (item.DataItem<int16>())
            | DType.Int32 -> box (item.DataItem<int32>())
            | DType.Int64 -> box (item.DataItem<int64>())
            | DType.Float32 -> box (item.DataItem<float32>())
            | DType.Float64 -> box (item.DataItem<double>())
            | _ -> failwith "Torch GetItem TBD type"
        obj

    member t.ToValuesTyped<'T, 'T2>(conv) : obj =
        match t.Shape with
        | [|  |] -> t.GetItem()
        | [| d0 |] -> upcast Array.init<'T> d0 (fun i -> tt.[int64 i].DataItem<'T2>() |> conv)
        | [| d0; d1 |] -> upcast Array2D.init<'T> d0 d1 (fun i j -> tt.[int64 i, int64 j].DataItem<'T2>() |> conv)
        | [| d0; d1; d2 |]  -> upcast Array3D.init<'T> d0 d1 d2 (fun i j k -> tt.[int64 i, int64 j, int64 k].DataItem<'T2>() |> conv)
        | [| d0; d1; d2; d3 |]  -> upcast Array4D.init<'T> d0 d1 d2 d3 (fun i j k l -> tt.[int64 i, int64 j, int64 k, int64 l].DataItem<'T2>() |> conv)
        | _ -> failwithf "Cannot get array for Tensor dimensions > 4. Consider slicing the Tensor. Shape: %A" t.Shape

    override t.ToValues() =
        match dtype with 
        | DType.Bool -> t.ToValuesTyped<bool, bool>(id)
        | DType.Byte -> t.ToValuesTyped<byte, byte>(id)
        | DType.Int8 -> t.ToValuesTyped<sbyte, sbyte>(sbyte)
        | DType.Int16 -> t.ToValuesTyped<int16, int16>(id)
        | DType.Int32 -> t.ToValuesTyped<int32, int32>(id)
        | DType.Int64 -> t.ToValuesTyped<int64, int64>(id)
        | DType.Float32 -> t.ToValuesTyped<float32, float32>(id)
        | DType.Float64 -> t.ToValuesTyped<double, double>(id)
        | DType.Other _ -> failwith "Torch GetItem TBD other type"

    member t.ToRawData<'T>() : 'T[] =
        let data = tt.Data<'T>()
        let res = Array.zeroCreate<'T> (int32 tt.NumberOfElements)
        for i in 0 .. int32 tt.NumberOfElements - 1 do
            res.[i] <- data.[i]
        res

    member t.ToRawData() =
        match dtype with 
        | DType.Bool -> t.ToRawData<bool>() |> box
        | DType.Byte -> t.ToRawData<byte>() |> box
        | DType.Int8 -> t.ToRawData<sbyte>() |> box
        | DType.Int16 -> t.ToRawData<int16>() |> box
        | DType.Int32 -> t.ToRawData<int32>() |> box
        | DType.Int64 -> t.ToRawData<int64>() |> box
        | DType.Float32 -> t.ToRawData<float32>() |> box
        | DType.Float64 -> t.ToRawData<double>() |> box
        | DType.Other _ -> failwith "Torch GetItem TBD other type"

    override _.StackTs(tensors, dim) =
        let tts, shapes = tensors |> Array.map (fun t -> (t :?> TorchRawTensor).TorchTensor, t.Shape) |> Array.unzip
        let _n, _shape1, _shape2, newShape = Shape.checkCanStack shapes dim
        let result = tts.Stack(int64 dim)
        (tensors.[0] :?> TorchRawTensor).MakeLike(result, newShape)

    override t.UnstackT(dim) = 
        let shape = t.Shape
        let _shape1, _shape2, unstackedShape = Shape.checkCanUnstack shape dim
        let results = tt.Unbind(dim)
        results |> Array.map (fun rvalues -> t.MakeLike(rvalues, shape=unstackedShape))

    override t.CatTs(tensors, dim) = 
        let values, shapes = tensors |> Array.map (fun t -> t.TorchTensor, t.Shape) |> Array.unzip
        let _n, _shape1, _m2, _shape3, outShape = Shape.checkCanCat shapes dim
        let result = values.Cat(int64 dim)
        t.MakeLike(result, outShape)

    override t.SplitT(sizes, dim) =
        let shape = t.Shape
        let outShapes = Shape.checkCanSplit shape sizes dim
        let results = tt.SplitWithSizes(int64s sizes, dim)
        (results, outShapes) ||> Array.map2 (fun rvalues outShape -> 
            t.MakeLike(rvalues, shape=outShape))

    override t.TransposeT2() =
        Shape.checkCanTranspose t.Dim
        let newShape = Shape.computeTranspose t.Shape
        let result = tt.T()
        t.MakeLike(result, shape=newShape)

    override t.SqueezeT(dim) = 
        let shape = t.Shape
        let newShape = Shape.squeeze dim shape
        let mutable res = tt
        let mutable c = 0
        for i in 0 .. t.Dim - 1 do
            if shape.[i] = 1 && (dim = -1 || i = dim) then 
                res <- res.Squeeze(int64 c)
            else   
                c <- c + 1
        t.MakeLike(res, shape=newShape)

    override t.UnsqueezeT(dim) = 
        let outputShape = Shape.checkCanUnsqueeze dim t.Shape
        t.MakeLike(tt.Unsqueeze(int64 dim), shape=outputShape)

    override t.FlipT(dims:int[]) = 
        let result = tt.Flip(int64s dims)
        t.MakeLike(result)

    override t.DilateT(dilations:int[]) = 
        Shape.checkCanDilate t.Dim dilations
        let outputShape = Shape.dilated t.Shape dilations
        let shape4d = Array.append (Array.replicate (4 - t.Dim) 1) shape
        let dilations4d = Array.append (Array.replicate (4 - dilations.Length) 1) dilations
        let t4d = t.Expand(shape4d)
        let one = t.OneLike().TorchTensor
        let w2 = 
            let mutable w = t.ZerosLike(dilations4d).TorchTensor
            w.[0L,0L,0L,0L] <- one
            w

        match t.Dim with 
        | 1 ->
            let len0 = int64 shape.[0]
            let dilation0 = int64 dilations.[0]
            let res1 = t4d.TorchTensor.ConvTranspose1D(w2, stride=Nullable(dilation0))
            let lenOut = (len0 - 1L)*dilation0+1L 
            let res2 = res1.Slice(3L,0L,lenOut,1L)
            let res3 = res2.Reshape([| lenOut |])
            t.MakeLike(res3, outputShape)
        | 2 ->
            let len0 = int64 shape.[0]
            let len1 = int64 shape.[1]
            let dilation0 = int64 dilations.[0]
            let dilation1 = int64 dilations.[1]
            let res1 = t4d.TorchTensor.ConvTranspose2D(w2, strides= [| dilation0; dilation1 |])
            let lenOut0 = (len0 - 1L)*dilation0+1L 
            let lenOut1 = (len1 - 1L)*dilation1+1L 
            let res2 = res1.Slice(2L,0L,lenOut0,1L)
            let res3 = res2.Slice(3L,0L,lenOut1,1L)
            let res4 = res3.Reshape([| lenOut0; lenOut1 |])
            t.MakeLike(res4, outputShape)
        | 3 ->
            let len0 = int64 shape.[0]
            let len1 = int64 shape.[1]
            let len2 = int64 shape.[2]
            let dilation0 = int64 dilations.[0]
            let dilation1 = int64 dilations.[1]
            let dilation2 = int64 dilations.[2]
            if dilation0 <> 1L then 
                failwith "DilateT 3D not functioning correctly in LibTorch - RuntimeError: expected stride to be a single integer value or a list of 2 values to match the convolution dimensions, but got stride=[2, 2, 2]"
            // let res1 =  t4d.TorchTensor.ConvTranspose3D(w2, strides= [| dilation0; dilation1; dilation2 |])
            let results = 
                [| for i in 0 .. shape.[0]-1 do
                      let slice = t4d.TorchTensor.Slice(1L,int64 i, int64 (i+1), 1L)
                      yield slice.ConvTranspose2D(w2, strides= [| dilation1; dilation2 |]) |]
            let res1 = results.Cat(1L)

            let lenOut0 = (len0 - 1L)*dilation0+1L 
            let lenOut1 = (len1 - 1L)*dilation1+1L 
            let lenOut2 = (len2 - 1L)*dilation2+1L 
            let res2 = res1.Slice(1L,0L,lenOut0,1L)
            let res3 = res2.Slice(2L,0L,lenOut1,1L)
            let res4 = res3.Slice(3L,0L,lenOut2,1L)
            let res5 = res4.Reshape([| lenOut0; lenOut1; lenOut2 |])
            t.MakeLike(res5, outputShape)
        | 4 ->
            let len0 = int64 shape.[0]
            let len1 = int64 shape.[1]
            let len2 = int64 shape.[2]
            let len3 = int64 shape.[3]
            let dilation0 = int64 dilations.[0]
            let dilation1 = int64 dilations.[1]
            let dilation2 = int64 dilations.[2]
            let dilation3 = int64 dilations.[3]
            if dilation0 <> 1L || dilation1 <> 1L then 
                failwith "DilateT 4D not easy in LibTorch unles dilation0 and dilation1 both 1"
            let res1 = 
                [| for i in 0 .. shape.[0]-1 do
                      [| for j in 0 .. shape.[1]-1 do
                            let slice = t4d.TorchTensor.Slice(0L,int64 i, int64 (i+1), 1L).Slice(1L,int64 j, int64 (j+1), 1L)
                            slice.ConvTranspose2D(w2, strides= [| dilation2; dilation3 |]) |].Cat(1L) |].Cat(0L)

            let lenOut0 = (len0 - 1L)*dilation0+1L 
            let lenOut1 = (len1 - 1L)*dilation1+1L 
            let lenOut2 = (len2 - 1L)*dilation2+1L 
            let lenOut3 = (len3 - 1L)*dilation3+1L 
            let res2 = res1.Slice(0L,0L,lenOut0,1L)
            let res3 = res2.Slice(1L,0L,lenOut1,1L)
            let res4 = res3.Slice(2L,0L,lenOut2,1L)
            let res5 = res4.Slice(3L,0L,lenOut3,1L)
            let res5 = res5.Reshape([| lenOut0; lenOut1; lenOut2; lenOut3 |])
            t.MakeLike(res5, outputShape)
        | _ ->
            failwith "DilateT > 3D not available in LibTorch"

    override t.UndilateT(dilations:int[]) =
        let outputShape = Shape.undilatedShape t.Shape dilations
        let mutable res = tt
        for d in 0 .. dilations.Length - 1 do
            res <- res.Slice(int64 d, 0L, int64 shape.[d], int64 dilations.[d])
        t.MakeLike(res, outputShape)

    override t.GatherT(dim:int, indices) =
        Shape.checkCanGather t.Shape dim indices.Shape indices.DType
        if indices.DType <> DType.Int32 then opNotSupported "Gather (indices must currently be int32 tensors in DiffSharp" indices.DType

        // NOTE: DiffSharp currently expects indices as an Int32 tensor, Torch wants Int64
        let indices = indices.Cast(DType.Int64)
        let res = t.TorchTensor.Gather(int64 dim, indices.TorchTensor)
        t.MakeLike(res, indices.Shape)

    override t.ViewT(shape:int[]) =
        Shape.checkCanView t.Shape shape
        t.MakeLike(tt.View(toTorchShape shape), shape=shape)

    override t.Cast(newDType: DType) =
        if newDType = t.DType then 
            upcast t
        else 
            let result = tt.ToType(toTorchType newDType)
            t.MakeLike(result, dtype=newDType)

    override t.MoveTo(device: Device) =
        if t.Device = device then (t :> _) else
        match device with 
        | Device.CPU -> t.MakeLike(t.TorchTensor.Cpu(), device=device)
        | Device.GPU -> t.MakeLike(t.TorchTensor.Cuda(), device=device)
        | _ -> invalidOp (sprintf "the device '%A' is not supported by the Torch backend" device)

    override _.Equals(t2:RawTensor) : bool = 
        if dtype = t2.DType then
            let r1 = (shape = t2.Shape)
            if not r1 then false else
            let tt2 = t2.TorchTensor
            let r2 = tt.Equal(tt2)
            r2
        else 
            opNotSupported2 "Equals" dtype t2.DType

    override t.AllClose(t2:RawTensor, relativeTolerance, absoluteTolerance) =
        if dtype = t2.DType then
            match dtype with 
            | DType.IntegralOrBool -> t.Equals(t2)
            | _ -> tt.AllClose(t2.TorchTensor, relativeTolerance, absoluteTolerance)
        else 
            opNotSupported2 "Equals" dtype t2.DType
        

    override t1.LtTT(t2) =
        let result = tt.Lt(t2.TorchTensor)
        t1.MakeLike(result, dtype=DType.Bool)

    override t1.GtTT(t2) =
        let result = tt.Gt(t2.TorchTensor)
        t1.MakeLike(result, dtype=DType.Bool)

    override t1.LeTT(t2) = 
        let result = tt.Le(t2.TorchTensor)
        t1.MakeLike(result, dtype=DType.Bool)

    override t1.GeTT(t2) = 
        let result = tt.Ge(t2.TorchTensor)
        t1.MakeLike(result, dtype=DType.Bool)

    override t1.EqTT(t2) = 
        let result = tt.Eq(t2.TorchTensor)
        t1.MakeLike(result, dtype=DType.Bool)

    override t1.NeqTT(t2) = 
        let result = tt.Ne(t2.TorchTensor)
        t1.MakeLike(result, dtype=DType.Bool)

    override t.MaxIndexT() = 
        let res = Array.zeroCreate<int64> t.Dim
        let idxs = Array.zeroCreate t.Dim
        let mutable values = tt
        for i = t.Dim - 1 downto 0 do 
            let (struct (values2, indexes)) = values.Max(int64 i)
            values <- values2
            idxs.[i] <- indexes
        for i = 0 to t.Dim - 1 do 
            let idx = idxs.[i]
            res.[i] <- 
                match i with 
                | 0 -> idx.DataItem<int64>()
                | 1 -> idx.[res.[0]].DataItem<int64>() 
                | 2 -> idx.[res.[0], res.[1]].DataItem<int64>() 
                | 3 -> idx.[res.[0], res.[1], res.[2]].DataItem<int64>() 
                | _ -> failwith "MaxIndexT > 4d nyi for torch"
        res |> Array.map int32

    // TODO: use Torch min operation
    override t.MinIndexT() = 
        match dtype with 
        | DType.Bool -> t.Cast(DType.Int8).MinIndexT() // TODO: could likely be improved
        | _ -> t.NegT().MaxIndexT()

    override t1.AddTT(t2) =
        let result = tt.Add(t2.TorchTensor) 
        t1.MakeLike(result)

    override t1.AddTT0(t2) =
        let t2v = t2.TorchTensor.Item()
        let result = tt.Add(t2v) 
        t1.MakeLike(result)

    override t1.AddT2T1(t2) = 
        let result = tt.Add(t2.TorchTensor) 
        t1.MakeLike(result)

    override t1.AddTTSlice(location:int[], t2) =
        Shape.checkCanAddSlice t1.Shape location t2.Shape
        let shape1 = t1.Shape
        let shape2 = t2.Shape
        let expandedShape2 = Shape.unsqueezeAs shape2 shape1
        let t2Expanded = t2.TorchTensor.Expand(toTorchShape expandedShape2)
        let res = tt.Clone()
        let mutable t1Slice = res // will share memory with res
        for d in 0 .. location.Length - 1 do 
            let len2 = expandedShape2.[d]
            if location.[d] <> 0 || len2 <> shape1.[d] then 
                t1Slice <- t1Slice.Narrow(int64 d, int64 location.[d], int64 len2)
        t1Slice.AddInPlace(t2Expanded) |> ignore
        t1.MakeLike(res)

    override t1.SubTT(t2) = 
        match dtype with 
        | DType.Bool -> opNotSupported2 "SubT" t1.DType t2.DType
        | _ ->
        let result = tt.Sub(t2.TorchTensor)
        t1.MakeLike(result)

    override t1.SubT0T(t2) =
        let t1v = t1.TorchTensor.Item()
        let result = t1v - t2.TorchTensor
        (t2 :?> TorchRawTensor).MakeLike(result)

    override t1.SubTT0(t2) = 
        let t2v = t2.TorchTensor.Item()
        let result = tt.Sub(t2v)
        t1.MakeLike(result)

    override t1.MulTT(t2) = 
        let result = tt.Mul(t2.TorchTensor)
        t1.MakeLike(result)

    override t1.MulTT0(t2) = 
        let t2v = t2.TorchTensor.Item()
        let result = tt.Mul(t2v)
        t1.MakeLike(result)

    override t1.DivTT(t2) = 
        match dtype with 
        | DType.Bool -> opNotSupported2 "DivTT" t1.DType t2.DType
        | _ ->
        let result = tt.Div(t2.TorchTensor)
        t1.MakeLike(result)

    override t1.DivT0T(t2) =
        match dtype with 
        | DType.Bool -> opNotSupported2 "DivTT" t1.DType t2.DType
        | _ ->
        let t1v = t1.TorchTensor.Item()
        let result = t1v / t2.TorchTensor
        (t2 :?> TorchRawTensor).MakeLike(result)

    override t1.DivTT0(t2) = 
        match dtype with 
        | DType.Bool -> opNotSupported2 "DivTT" t1.DType t2.DType
        | _ ->
        let t2v = t2.TorchTensor.Item()
        let result = tt.Div(t2v)
        t1.MakeLike(result)

    override t1.PowTT(t2) =
        match dtype with 
        | DType.IntegralOrBool -> opNotSupported "PowTT" dtype
        | _ -> 
        let result = tt.Pow(t2.TorchTensor)
        t1.MakeLike(result)

    override t1.PowT0T(t2) = 
        match dtype with 
        | DType.IntegralOrBool -> opNotSupported "PowT0T" dtype
        | _ -> 
        let result = t1.Expand(t2.Shape).TorchTensor.Pow(t2.TorchTensor)
        (t2 :?> TorchRawTensor).MakeLike(result)

    override t1.PowTT0(t2) =
        match dtype with 
        | DType.IntegralOrBool -> opNotSupported "PowTT0" dtype
        | _ -> 
        let t2v = t2.TorchTensor.Item()
        let result = tt.Pow(t2v)
        t1.MakeLike(result)

    override t1.MatMulT2T2(t2) = 
        match dtype with 
        | DType.Bool -> opNotSupported2 "MatMulT2T2" t1.DType t2.DType
        | _ ->  
        Shape.checkCanMatmul t1.Shape t2.Shape
        let result = tt.Mm(t2.TorchTensor)
        t1.MakeLike(result, [| t1.Shape.[0]; t2.Shape.[1] |])

    override t1.Conv1D(t2, stride, padding) = // TODO: bias, dilation and groups
        let _batchSize, _inputChannels, _kernelSize, _outputChannels, _outputSize, outputShape = Shape.checkCanConv1d t1.DType t2.DType t1.Shape t2.Shape stride padding 1
        match t1.DType, t2.DType with 
        | DType.Bool, _ | _, DType.Bool -> opNotSupported2 "Conv1D" t1.DType t2.DType
        | _ ->
        let resultt = t1.TorchTensor.Conv1D(t2.TorchTensor, stride=Nullable(int64 stride), padding=Nullable(int64 padding), dilation=Nullable(1L))
        t1.MakeLike(resultt, shape=outputShape)

    override t1.Conv2D(t2, strides, paddings) = // TODO: bias, dilation and groups
        let _batchSize, _inputChannels, _kernelDimensions, _outputDimensions, outputShape  = Shape.checkCanConv2d t1.DType t2.DType t1.Shape t2.Shape strides paddings [| 1;1 |]
        match t1.DType, t2.DType with 
        | DType.Bool, _ | _, DType.Bool -> opNotSupported2 "Conv2D" t1.DType t2.DType
        | _ ->
        let resultt = tt.Conv2D(t2.TorchTensor, strides=int64s strides, padding=int64s paddings)
        t1.MakeLike(resultt, shape=outputShape)

    override t1.Conv3D(t2, strides, paddings) = // TODO: bias, dilation and groups
        let _batchSize, _inputChannels, _kernelDimensions, _outputDimensions, outputShape = Shape.checkCanConv3d t1.DType t2.DType  t1.Shape t2.Shape strides paddings [| 1;1;1 |]
        match t1.DType, t2.DType with 
        | DType.Bool, _ | _, DType.Bool -> opNotSupported2 "Conv3D" t1.DType t2.DType
        | _ ->
        let resultt = tt.Conv3D(t2.TorchTensor, strides=int64s strides, padding=int64s paddings)
        t1.MakeLike(resultt, shape=outputShape)

    override t1.MaxPool1D(kernelSize, stride, padding) = 
        let _batchSize, _channels, _inputSize, _outputSize, outputShape = Shape.checkCanMaxpool1d t1.Shape kernelSize stride padding
        match t1.DType with 
        | DType.Bool | DType.Integral -> opNotSupported "MaxPool1D" t1.DType
        | _ ->
        let struct (resultt, indicest) = tt.MaxPool1DWithIndices(int64 kernelSize, stride=Nullable(int64 stride), padding=Nullable(int64 padding), dilation=Nullable(1L))
        // NOTE: DiffSharp currently expects indices as an Int32 tensor
        let indices = t1.MakeLike(indicest, shape=outputShape, dtype=DType.Int64).Cast(DType.Int32)
        let result = t1.MakeLike(resultt, shape=outputShape)
        result, indices

    override t1.MaxPool2D(kernelSize, strides, paddings) = 
        let _batchSize, _channels, _inputDimensions, _kernelDimensions, _outputDimensions, outputShape = Shape.checkCanMaxpool2d t1.Shape kernelSize strides paddings
        match t1.DType with 
        | DType.Bool | DType.Integral -> opNotSupported "MaxPool2D" t1.DType
        | _ ->
        let struct (resultt, indicest) = tt.MaxPool2DWithIndices(int64s kernelSize, strides=int64s strides, padding=int64s paddings)
        // NOTE: DiffSharp currently expects indices as an Int32 tensor, Torch wants Int64
        let indices = t1.MakeLike(indicest, shape=outputShape, dtype=DType.Int64).Cast(DType.Int32)
        let result = t1.MakeLike(resultt, shape=outputShape)
        result, indices

    override t1.MaxPool3D(kernelSize, strides, paddings) = 
        let _batchSize, _channels, _inputDimensions, _kernelDimensions, _outputDimensions, outputShape = Shape.checkCanMaxpool3d t1.Shape kernelSize strides paddings
        match t1.DType with 
        | DType.Bool | DType.Integral -> opNotSupported "MaxPool3D" t1.DType 
        | _ ->
        let struct (resultt, indicest) = tt.MaxPool3DWithIndices(int64s kernelSize, strides=int64s strides, padding=int64s paddings)
        
        // NOTE: DiffSharp currently expects indices as an Int32 tensor
        let indices = t1.MakeLike(indicest, shape=outputShape, dtype=DType.Int64).Cast(DType.Int32)
        let result = t1.MakeLike(resultt, shape=outputShape)
        result, indices

    override t1.MaxUnpool1D(indices, outputSize) = 
        // NOTE: LibTorch has no torch::max_unpool1d and so TorchSharp has Tensor.MaxUnpool1D
        // So use MaxUnpool2D instead
        //let batchSize, channels, _inputSize, _outputShape = Shape.computeMaxUnpool1d t1.Shape outputSize
        let t1X = t1.UnsqueezeT(2)
        let indicesX = indices.UnsqueezeT(2)
        let resulttX = t1X.MaxUnpool2D(indicesX, [| outputSize.[0]; outputSize.[1]; 1; outputSize.[2] |])
        let resultt = resulttX.SqueezeT(2)
        resultt

    override t1.MaxUnpool2D(indices, outputSize) = 
        let _batchSize, _channels, _inputDimensions, outputShape = Shape.checkCanMaxunpool2d t1.Shape indices.DType indices.Shape outputSize
        match t1.DType with 
        | DType.Bool | DType.Integral -> opNotSupported "MaxUnpool2D" t1.DType 
        | _ ->
        // NOTE: DiffSharp currently expects indices as an Int32 tensor
        let indices = indices.Cast(DType.Int64)

        // note, LibTorch only wants the last two elements of the output size passsed in
        // "There should be exactly two elements (height, width) in output_size (max_unpooling2d_shape_check at ...)"
        let outputSize = outputSize.[2..3]
        
        // TODO: consider switching to the torch::nn module for MaxUnpool2d

        let resultt = tt.MaxUnpool2D(indices.TorchTensor, int64s outputSize)
        t1.MakeLike(resultt, shape=outputShape)

    override t1.MaxUnpool3D(indices, outputSize) = 
        let _batchSize, _channels, _inputDimensions, outputShape = Shape.checkCanMaxunpool3d t1.Shape indices.DType indices.Shape outputSize
        match t1.DType with 
        | DType.Bool | DType.Integral -> opNotSupported "MaxUnpool3D" t1.DType 
        | _ ->
        // NOTE: DiffSharp currently expects indices as an Int32 tensor
        let indices = indices.Cast(DType.Int64)

        // note, LibTorch only wants the last three elements of the output size passsed in
        // "There should be exactly three elements (depth, height, width) in output_size (max_unpooling3d_shape_check at ..\..\aten\src\ATen\native\MaxUnpooling.cpp:231)"
        let outputSize = outputSize.[2..4]
        
        // NOTE: strides and padding must always be specified for torch::max_unpool3d C++ entry
        // TODO: consider switching to the torch::nn module for MaxUnpool
        let strides = outputSize |> Array.map (fun _ -> 1L)
        let padding = outputSize |> Array.map (fun _ -> 0L)
        let resultt = tt.MaxUnpool3D(indices.TorchTensor, int64s outputSize, strides, padding)
        t1.MakeLike(resultt, shape=outputShape)

    override t.SumT2Dim0() =
        let result = tt.Sum([| 0L |], ``type``= Nullable(tt.Type))
        let resultShape = [|t.Shape.[1]|]
        t.MakeLike(result, shape=resultShape)

    override t.NegT() =
        match dtype with 
        | DType.Bool -> opNotSupported "NegT" t.DType
        | _ ->  t.MakeLike(-tt)

    override t.SumT(?resultType) =
        let typeArg = match resultType with None -> Nullable() | Some dt -> Nullable(toTorchType dt)
        let outType = match resultType with None -> dtype.SummationType | Some dt -> dt
        t.MakeLike(tt.Sum(typeArg), shape=Shape.scalar, dtype=outType)

    override t.SignT() =
        t.MakeLike(tt.Sign())

    override t.FloorT() =
        match dtype with 
        | DType.IntegralOrBool -> opNotSupported "FloorT" t.DType
        | _ ->  t.MakeLike(tt.Floor())

    override t.CeilT() =
        match dtype with 
        | DType.IntegralOrBool -> opNotSupported "CeilT" t.DType
        | _ ->  t.MakeLike(tt.Ceil())

    override t.RoundT() =
        match dtype with 
        | DType.IntegralOrBool -> opNotSupported "RoundT" t.DType
        | _ ->  t.MakeLike(tt.Round())

    override t.AbsT() = 
        match dtype with 
        | DType.Bool -> opNotSupported "AbsT" t.DType
        | DType.Int8 -> t.Cast(DType.Int32).AbsT().Cast(DType.Int8) // TODO: there is odd behaviour from torch for relu on int8, may have been fixed in later version?
        | _ -> t.MakeLike(tt.Abs ())

    override t.SoftplusT() = 
        match dtype with 
        | DType.IntegralOrBool -> opNotSupported "SoftplusT" t.DType
        | _ -> t.MakeLike(tt.Softplus())

    override t.ReluT() =
        match dtype with 
        | DType.Bool -> opNotSupported "ReluT" t.DType
        | DType.Int8 -> t.Cast(DType.Int32).ReluT().Cast(DType.Int8) // TODO: there is odd behaviour from torch for relu on int8, may have been fixed in later version?
        | _ ->   t.MakeLike(tt.Relu())

    override t.SigmoidT() =
        match dtype with 
        | DType.IntegralOrBool -> opNotSupported "SigmoidT" t.DType
        | _ ->  t.MakeLike(tt.Sigmoid())

    override t.ExpT() =
        match dtype with 
        | DType.IntegralOrBool -> opNotSupported "ExpT" t.DType
        | _ ->  t.MakeLike(tt.Exp())

    override t.LogT() =
        match dtype with 
        | DType.IntegralOrBool -> opNotSupported "LogT" t.DType
        | _ ->  t.MakeLike(tt.Log())

    override t.Log10T() =
        match dtype with 
        | DType.IntegralOrBool -> opNotSupported "Log10T" t.DType
        | _ ->   t.MakeLike(tt.Log10())

    override t.SqrtT() =
        match dtype with 
        | DType.IntegralOrBool -> opNotSupported "SqrtT" t.DType
        | _ ->  t.MakeLike(tt.Sqrt())

    override t.SinT() =
        match dtype with 
        | DType.IntegralOrBool -> opNotSupported "SinT" t.DType
        | _ ->  t.MakeLike(tt.Sin())

    override t.CosT() =
        match dtype with 
        | DType.IntegralOrBool -> opNotSupported "CosT" t.DType
        | _ ->  t.MakeLike(tt.Cos())

    override t.TanT() =
        match dtype with 
        | DType.IntegralOrBool -> opNotSupported "TanT" t.DType
        | _ ->  t.MakeLike(tt.Tan())

    override t.SinhT() =
        match dtype with 
        | DType.IntegralOrBool -> opNotSupported "SinhT" t.DType
        | _ ->  t.MakeLike(tt.Sinh())

    override t.CoshT() =
        match dtype with 
        | DType.IntegralOrBool -> opNotSupported "CoshT" t.DType
        | _ ->  t.MakeLike(tt.Cosh())

    override t.TanhT() =
        match dtype with 
        | DType.IntegralOrBool -> opNotSupported "TanhT" t.DType
        | _ ->  t.MakeLike(tt.Tanh())

    override t.AsinT() =
        match dtype with 
        | DType.IntegralOrBool -> opNotSupported "AsinT" t.DType
        | _ ->  t.MakeLike(tt.Asin())

    override t.AcosT() =
        match dtype with 
        | DType.IntegralOrBool -> opNotSupported "AcosT" t.DType
        | _ ->  t.MakeLike(tt.Acos())

    override t.AtanT() =
        match dtype with 
        | DType.IntegralOrBool -> opNotSupported "AtanT" t.DType
        | _ ->  t.MakeLike(tt.Atan())

    new (info: System.Runtime.Serialization.SerializationInfo, _context: System.Runtime.Serialization.StreamingContext) =
        let device = info.GetValue("device", typeof<Device>) :?> Device
        let dtype = info.GetValue("dtype", typeof<DType>) :?> DType
        let shape = info.GetValue("shape", typeof<Shape>) :?> Shape
        let tt =
            match dtype with 
            | DType.Bool -> 
                let data = info.GetValue("data", typeof<bool[]>)  :?> bool[]
                BoolTensor.From (data, toTorchShape shape) 
            | DType.Byte -> 
                let data = info.GetValue("data", typeof<byte[]>)  :?> byte[]
                ByteTensor.From (data, toTorchShape shape) 
            | DType.Int8 -> 
                let data = info.GetValue("data", typeof<sbyte[]>)  :?> sbyte[]
                SByteTensor.From (data, toTorchShape shape) 
            | DType.Int16 -> 
                let data = info.GetValue("data", typeof<int16[]>)  :?> int16[]
                ShortTensor.From (data, toTorchShape shape) 
            | DType.Int32 -> 
                let data = info.GetValue("data", typeof<int32[]>)  :?> int32[]
                IntTensor.From (data, toTorchShape shape) 
            | DType.Int64 -> 
                let data = info.GetValue("data", typeof<int64[]>)  :?> int64[]
                LongTensor.From (data, toTorchShape shape) 
            | DType.Float32 -> 
                let data = info.GetValue("data", typeof<float32[]>)  :?> float32[]
                FloatTensor.From (data, toTorchShape shape) 
            | DType.Float64 -> 
                let data = info.GetValue("data", typeof<double[]>)  :?> double[]
                DoubleTensor.From (data, toTorchShape shape) 
            | DType.Other _ -> failwith "deserialize other type in torch nyi"

        TorchRawTensor(tt, shape, dtype, device, Immutable)

    interface System.Runtime.Serialization.ISerializable with

        //[SecurityPermissionAttribute(SecurityAction.Demand,  SerializationFormatter = true)]
        member t.GetObjectData(info, _context) =
            info.AddValue("device", device)
            info.AddValue("dtype", dtype)
            info.AddValue("shape", shape)
            info.AddValue("data", t.ToRawData())

/// The concrete implementation of BackendStatics for Float32 data.
type TorchStatics<'T, 'T2>
       (dtype: DType, conv: 'T -> 'T2,
        from0: 'T2 -> TorchTensor,
        from: 'T2[] * TorchShape -> TorchTensor,
        zero: 'T, one: 'T,
        zeros: TorchShape  * string -> TorchTensor,
        ones: TorchShape  * string -> TorchTensor,
        random: TorchShape  * string -> TorchTensor,
        randomN: TorchShape  * string -> TorchTensor,
        randomIntegers: TorchShape * int * int * string -> TorchTensor,
        valueFromObj: obj -> 'T,
        scalarFromConvValue: 'T2 -> Scalar) = 

    inherit BackendStatics()

    override _.Seed(seed) = Torch.SetSeed(int64 seed)
    override _.Zero(device) = TorchRawTensor(from0(conv(zero)), Shape.scalar, dtype, device) :> _ 
    override _.One(device) = TorchRawTensor(from0(conv(one)), Shape.scalar, dtype, device) :> _
    override _.Zeros(shape:int[], device, mutability) = TorchRawTensor(zeros(toTorchShape shape, toTorchDevice device), shape, dtype, device, mutability) :> _
    override _.Ones(shape:int[], device) = TorchRawTensor(ones(toTorchShape shape, toTorchDevice device), shape, dtype, device) :> _
    override _.Random(shape:int[], device) = TorchRawTensor(random(toTorchShape shape, toTorchDevice device), shape, dtype, device) :> _
    override _.RandomNormal(shape:int[], device) = TorchRawTensor(randomN(toTorchShape shape, toTorchDevice device), shape, dtype, device) :> _
    override _.RandomInt(shape, low, high, device) = TorchRawTensor(randomIntegers(toTorchShape shape, low, high, toTorchDevice device), shape, dtype, device) :> _

    override _.Full(shape:int[], value:obj, device) =
        let t = zeros(toTorchShape shape, toTorchDevice device)
        t.FillInPlace(scalarFromConvValue (conv (valueFromObj value))) |> ignore
        TorchRawTensor(t, shape, dtype, device) :> _

    override ts.CreateFromFlatArray(values:Array, shape, device) =
        let values = values :?> 'T[] |> Array.map conv 
        let t = 
            match shape with 
            | [| |] -> from0(values.[0])
            | _ -> from (values, toTorchShape shape)
        let tt = 
           match device with 
           | Device.CPU -> t
           | Device.GPU -> t.Cuda()
           | Device.Other _ -> failwith "device extensibility not available in Torch backend as yet"
        TorchRawTensor(tt, shape, dtype, device) :> _

/// The concrete implementation of BackendStatics for Bool  data.
type TorchFloat32Statics() = 

    inherit TorchStatics<single, single>(DType.Float32, id, 
        (fun v -> FloatTensor.From(v)), 
        (fun (data, shape) -> FloatTensor.From(data, shape)), 
        0.0f, 1.0f, 
        (fun (shape, device) -> FloatTensor.Zeros(shape, device=device)), 
        (fun (shape, device) -> FloatTensor.Ones(shape, device=device)), 
        (fun (shape, device) -> FloatTensor.Random(shape, device=device)), 
        (fun (shape, device) -> FloatTensor.RandomN(shape, device=device)), 
        (fun (shape, low, high, device) -> FloatTensor.RandomIntegers(int64 (high-low), shape, device=device).AddInPlace((float low).ToScalar())), 
        System.Convert.ToSingle, 
        Scalar.op_Implicit)

type TorchFloat64Statics() = 

    inherit TorchStatics<double, double>(DType.Float64, id, 
        (fun v -> DoubleTensor.From(v)), 
        (fun (data, shape) -> DoubleTensor.From(data, shape)), 
        0.0, 1.0, 
        (fun (shape, device) -> DoubleTensor.Zeros(shape, device=device)), 
        (fun (shape, device) -> DoubleTensor.Ones(shape, device=device)), 
        (fun (shape, device) -> DoubleTensor.Random(shape, device=device)), 
        (fun (shape, device) -> DoubleTensor.RandomN(shape, device=device)), 
        (fun (shape, low, high, device) -> DoubleTensor.RandomIntegers(int64 (high-low), shape, device=device).AddInPlace((double low).ToScalar())), 
        System.Convert.ToDouble, 
        Scalar.op_Implicit)

type TorchInt8Statics() = 

    inherit TorchStatics<sbyte, sbyte>(DType.Int8, sbyte,
        (fun v -> SByteTensor.From(v)), 
        (fun (data, shape) -> SByteTensor.From(data, shape)), 
        0y, 1y,
        (fun (shape, device) -> SByteTensor.Zeros(shape, device=device)), 
        (fun (shape, device) -> SByteTensor.Ones(shape, device=device)), 
        (fun _ -> opNotSupported "Random" DType.Int8), 
        (fun _ -> opNotSupported "RandomNormal" DType.Int8), 
        (fun (shape, low, high, device) -> SByteTensor.RandomIntegers(int64 (high-low), shape, device=device).AddInPlace((sbyte low).ToScalar())), 
        System.Convert.ToSByte, 
        Scalar.op_Implicit)

type TorchInt16Statics() = 

    inherit TorchStatics<int16, int16>(DType.Int16, int16, 
        (fun v -> ShortTensor.From(v)), 
        (fun (data, shape) -> ShortTensor.From(data, shape)), 
        0s, 1s,
        (fun (shape, device) -> ShortTensor.Zeros(shape, device=device)), 
        (fun (shape, device) -> ShortTensor.Ones(shape, device=device)), 
        (fun _ -> opNotSupported "Random" DType.Int16), 
        (fun _ -> opNotSupported "RandomNormal" DType.Int16), 
        (fun (shape, low, high, device) -> ShortTensor.RandomIntegers(int64 (high-low), shape, device=device).AddInPlace((int16 low).ToScalar())), 
        System.Convert.ToInt16, 
        Scalar.op_Implicit)

type TorchInt32Statics() = 

    inherit TorchStatics<int32, int32>(DType.Int32, int32, 
        (fun v -> IntTensor.From(v)), 
        IntTensor.From, 
        0, 1,
        (fun (shape, device) -> IntTensor.Zeros(shape, device=device)), 
        (fun (shape, device) -> IntTensor.Ones(shape, device=device)), 
        (fun _ -> opNotSupported "Random" DType.Int32), 
        (fun _ -> opNotSupported "RandomNormal" DType.Int32), 
        (fun (shape, low, high, device) -> IntTensor.RandomIntegers(int64 (high-low), shape, device=device).AddInPlace((int32 low).ToScalar())), 
        System.Convert.ToInt32, 
        Scalar.op_Implicit)

type TorchInt64Statics() = 

    inherit TorchStatics<int64, int64>(DType.Int64, int64, 
        (fun v -> LongTensor.From(v)), 
        (fun (data, shape) -> LongTensor.From(data, shape)), 
        0L, 1L,
        (fun (shape, device) -> LongTensor.Zeros(shape, device=device)), 
        (fun (shape, device) -> LongTensor.Ones(shape, device=device)), 
        (fun _ -> opNotSupported "Random" DType.Int64), 
        (fun _ -> opNotSupported "RandomNormal" DType.Int64), 
        (fun (shape, low, high, device) -> LongTensor.RandomIntegers(int64 (high-low), shape, device=device).AddInPlace((int64 low).ToScalar())), 
        System.Convert.ToInt64, 
        Scalar.op_Implicit)

type TorchBoolStatics() = 

    inherit TorchStatics<bool, bool>(DType.Bool, id, 
        (fun v -> BoolTensor.From(v)), 
        (fun (data, shape) -> BoolTensor.From(data, shape)), 
        false, true,
        (fun (shape, device) -> BoolTensor.Zeros(shape, device=device)), 
        (fun (shape, device) -> BoolTensor.Ones(shape, device=device)), 
        (fun _ -> opNotSupported "Random" DType.Bool), 
        (fun _ -> opNotSupported "RandomNormal"  DType.Bool), 
        (fun (shape, low, high, device) -> BoolTensor.RandomIntegers(min 2L (int64 (high-low)), shape, device=device).AddInPlace((low > 0).ToScalar())), 
        System.Convert.ToBoolean, 
        Scalar.op_Implicit)

type TorchByteStatics() = 

    inherit TorchStatics<byte, byte>(DType.Byte, id, 
        (fun v -> ByteTensor.From(v)), 
        (fun (data, shape) -> ByteTensor.From(data, shape)), 
        0uy, 1uy,
        (fun (shape, device) -> ByteTensor.Zeros(shape, device=device)), 
        (fun (shape, device) -> ByteTensor.Ones(shape, device=device)), 
        (fun _ -> opNotSupported "Random" DType.Byte), 
        (fun _ -> opNotSupported "RandomNormal"  DType.Byte), 
        (fun (shape, low, high, device) -> ByteTensor.RandomIntegers(int64 (high-low), shape, device=device).AddInPlace((byte low).ToScalar())), 
        System.Convert.ToByte, 
        Scalar.op_Implicit)
