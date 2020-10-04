namespace rec DiffSharp.Backends.Symbolic

open System
open DiffSharp
open DiffSharp.Backends
open DiffSharp.Util

#nowarn "77" // use of op_Explicit

type SymbolicShape = Shape

type SymbolicTensor(shape: SymbolicShape, dtype: Dtype, device: Device) =
    inherit RawTensor()

    let sample =
        match dtype with 
        | Float32 -> box 0.0f
        | Float64 -> box 0.0
        | Int8 -> box 0y
        | Byte -> box 0uy
        | Int16 -> box 0s
        | Int32 -> box 0
        | Int64 -> box 0L
        | Bool -> box false
        | Sym _ -> box 0.0

    member t.MakeLike(?shape: Shape, ?device: Device, ?dtype) =
        SymbolicTensor(defaultArg shape t.Shape, defaultArg dtype t.Dtype, defaultArg device t.Device) :> RawTensor

    override _.Shape = shape
    override _.Dim = shape.Length
    override _.Nelement = Shape.nelements shape
    override _.Dtype = dtype
    override _.Device = device
    override _.DeviceType = device.DeviceType
    override _.Handle = box 0
    override _.Backend = Backend.Symbolic

    override t.GetItem(_indexes) =
        printfn "GetItem not available for symbolic"
        sample

    override t.GetSlice(fullBounds:Dim[,]) =
        let shape = Shape.checkCanGetSlice t.Shape fullBounds
        t.MakeLike(shape)

    override t.Clone() = t :> RawTensor

    override _.ComputeHash() = hash shape 
    
    override t.Expand(newShape) = t.MakeLike(newShape)

    override t.ToValues() =
        printfn "ToValues not available for symbolic"
        match t.Dim with
        | 0 -> box sample
        | _ -> 
            let dims = t.Shape.Dims |> Array.map (fun d -> d.ValueOrOne)
            match dtype with 
            | Float32 -> arrayND dims (fun _ -> 0.0f)
            | Float64 -> arrayND dims (fun _ -> 0.0)
            | Int8 -> arrayND dims (fun _ -> 0y)
            | Byte -> arrayND dims (fun _ -> 0uy)
            | Int16 -> arrayND dims (fun _ -> 0s)
            | Int32 -> arrayND dims (fun _ -> 0)
            | Int64 -> arrayND dims (fun _ -> 0L)
            | Bool -> arrayND dims (fun _ -> false)
            | Sym _ -> 
                printfn "ToValues not available for symbolic dtype"
                arrayND dims (fun _ -> 0)
            

    override _.StackTs(tensors, dim) =
        let shapes = tensors |> Array.map (fun t -> t.Shape)
        let _, _, _, newShape = Shape.checkCanStack shapes dim
        (tensors.[0] :?> SymbolicTensor).MakeLike(newShape)

    override t.UnstackT(dim) =
        let shape = t.Shape
        let _, _, unstackedShape = Shape.checkCanUnstack shape dim
        let n = shape.[dim].Value // the value must be known to do an unstack
        Array.init n (fun _ -> t.MakeLike(unstackedShape))

    override t.CatTs(tensors, dim) =
        let shapes = tensors |> Array.map (fun t -> t.Shape)
        let _, _, _, _, outShape = Shape.checkCanCat shapes dim
        t.MakeLike(outShape)

    override t.SplitT(sizes, dim) =
        let shape = t.Shape
        let outShapes = Shape.checkCanSplit shape sizes dim
        outShapes |> Array.map (fun outShape -> t.MakeLike(outShape))

    override t.TransposeT(dim0, dim1) =
        Shape.checkCanTranspose t.Shape dim0 dim1
        let shape = Array.copy t.Shape.Dims
        shape.[dim0] <- t.Shape.[dim1]
        shape.[dim1] <- t.Shape.[dim0]
        t.ZerosLike(Shape shape)

    override t.TransposeT2() =
        Shape.checkCanTranspose2d t.Dim
        t.MakeLike(Shape [| t.Shape.[1]; t.Shape.[0]|])

    override t.SqueezeT(dim) =
        t.MakeLike(Shape.squeeze dim t.Shape)

    override t.UnsqueezeT(dim) =
        let outputShape = Shape.checkCanUnsqueeze dim t.Shape
        t.MakeLike(outputShape)

    override t.FlipT(dims:int[]) =
        Shape.checkCanFlip t.Dim dims
        t :> RawTensor

    override t.DilateT(dilations:int[]) =
        Shape.checkCanDilate t.Dim dilations
        t.MakeLike(Shape.dilated t.Shape dilations) 

    override t.UndilateT(dilations:int[]) =
        t.MakeLike(Shape.undilatedShape t.Shape dilations)

    override t.GatherT(dim:int, indices) =
        Shape.checkCanGather t.Shape dim indices.Shape indices.Dtype
        t.MakeLike(indices.Shape) 

    override t.ViewT(shape:Shape) =
        Shape.checkCanView t.Shape shape
        t.MakeLike(shape)

    override t.Cast(dtype: Dtype) = t.MakeLike(dtype=dtype)

    override t.MoveTo(device: Device) = t.MakeLike(shape, device=device)

    override t1.Equals(_t2:RawTensor) : bool =  printfn "Equals"; false
    override t1.AllClose(_t2:RawTensor, _relativeTolerance, _absoluteTolerance) = printfn "AllClose"; false
    override t.ClampT(_low, _high) = t :> _ 
    override t.SoftplusT() = t :> _ 
    override t1.LtTT(_t2) = t1.MakeLike(dtype=Bool)
    override t1.GtTT(_t2) = t1.MakeLike(dtype=Bool)
    override t1.LeTT(_t2) = t1.MakeLike(dtype=Bool)
    override t1.GeTT(_t2) = t1.MakeLike(dtype=Bool)
    override t1.EqTT(_t2) = t1.MakeLike(dtype=Bool)
    override t1.NeqTT(_t2) = t1.MakeLike(dtype=Bool)
    override t.MaxIndexT() = printfn "MaxIndexT"; Array.zeroCreate t.Dim
    override t.MinIndexT() = printfn "MinIndexT"; Array.zeroCreate t.Dim
    override t1.AddTT(_t2) = t1.MakeLike()
    override t1.AddTT0(_t2) = t1.MakeLike()
    override t1.AddT2T1(_t2) = t1.MakeLike()
    override t1.AddTTSlice(_location:Dim[], _t2) = t1.MakeLike()
    override t1.SubTT(_t2) = t1.MakeLike()
    override t1.SubT0T(t2) = t2
    override t1.SubTT0(_t2) = t1.MakeLike()
    override t1.MulTT(_t2) = t1.MakeLike()
    override t1.MulTT0(_t2) = t1.MakeLike()
    override t1.DivTT(_t2) = t1.MakeLike()
    override t1.DivT0T(t2) = t2
    override t1.DivTT0(_t2) = t1 :>_
    override t1.PowTT(_t2) = t1 :>_
    override t1.PowT0T(t2) = t2
    override t1.PowTT0(_t2) = t1 :>_

    override t1.MatMulT2T2(t2) = 
        Shape.checkCanMatmul t1.Shape t2.Shape
        let t1rows, _t1cols = t1.Shape.[0], t1.Shape.[1]
        let _t2rows, t2cols = t2.Shape.[0], t2.Shape.[1]
        t1.MakeLike(Shape [| t1rows; t2cols |])

    override t1.MaxPool1D(kernelSize, stride, padding) = 
        let _, _, _, _, outputShape = Shape.checkCanMaxpool1d t1.Dtype t1.Shape kernelSize stride padding
        t1.MakeLike(outputShape), t1.MakeLike(outputShape, dtype=Int32)

    override t1.MaxPool2D(kernelSize, stride, padding) =
        let _, _, _, _, _, outputShape = Shape.checkCanMaxpool2d t1.Dtype t1.Shape kernelSize stride padding
        t1.MakeLike(outputShape), t1.MakeLike(outputShape, dtype=Int32)

    override t1.MaxPool3D(kernelSize, stride, padding) =
        let _, _, _, _, _, outputShape = Shape.checkCanMaxpool3d t1.Dtype t1.Shape kernelSize stride padding
        t1.MakeLike(outputShape), t1.MakeLike(outputShape, dtype=Int32)

    override t1.MaxUnpool1D(indices, outputSize) =
        let _, _, _, outputShape = Shape.checkCanMaxunpool1d t1.Dtype t1.Shape indices.Dtype indices.Shape outputSize
        t1.MakeLike(outputShape)

    override t1.MaxUnpool2D(indices, outputSize) =
        let _, _, _, outputShape = Shape.checkCanMaxunpool2d t1.Dtype t1.Shape indices.Dtype indices.Shape outputSize
        t1.MakeLike(outputShape)

    override t1.MaxUnpool3D(indices, outputSize) =
        let _, _, _, outputShape = Shape.checkCanMaxunpool3d t1.Dtype t1.Shape indices.Dtype indices.Shape outputSize
        t1.MakeLike(outputShape)

    override t1.Conv1D(t2, stride, padding) = 
        let _, _, _, _, _, outputShape = Shape.checkCanConv1d t1.DeviceType t2.DeviceType t1.Dtype t2.Dtype t1.Shape t2.Shape stride padding 1
        t1.MakeLike(outputShape)

    override t1.Conv2D(t2, stride, padding) = 
        let _, _, _, _, outputShape = Shape.checkCanConv2d t1.DeviceType t2.DeviceType t1.Dtype t2.Dtype t1.Shape t2.Shape stride padding [|1;1|]
        t1.MakeLike(outputShape) 

    override t1.Conv3D(t2, stride, padding) = 
        let _, _, _, _, outputShape = Shape.checkCanConv3d t1.DeviceType t2.DeviceType t1.Dtype t2.Dtype t1.Shape t2.Shape stride padding [|1;1;1|]  
        t1.MakeLike(outputShape) 

    override t.NegT() = t :> _

    override t.SumT(resultType) =
        match resultType with 
        | None -> t.MakeLike(Shape.scalar)
        | Some dtype -> t.MakeLike(Shape.scalar, dtype=dtype)
    override t.SumT2Dim0() = t.MakeLike(Shape [|t.Shape.[1]|])
    override t.SignT() = t :> _ 
    override t.FloorT() = t :> _ 
    override t.CeilT() = t :> _ 
    override t.RoundT() = t :> _ 
    override t.AbsT() = t :> _ 
    override t.ReluT() = t :> _ 
    override t.SigmoidT() = t :> _ 
    override t.ExpT() = t :> _ 
    override t.LogT() = t :> _ 
    override t.Log10T() = t :> _ 
    override t.SqrtT() = t :> _ 
    override t.SinT() = t :> _ 
    override t.CosT() = t :> _ 
    override t.TanT() = t :> _ 
    override t.SinhT() = t :> _ 
    override t.CoshT() = t :> _ 
    override t.TanhT() = t :> _ 
    override t.AsinT() = t :> _ 
    override t.AcosT() = t :> _ 
    override t.AtanT() = t :> _ 
    override t.GetString() = sprintf "tensor(dtype=%A,shape=%A,device=%A)" dtype shape device

type SymbolicBackendStatics() = 

    inherit BackendStatics()

    override _.GetDevices(_deviceType) = [ Device.CPU; Device.GPU ]
    override _.IsDeviceTypeSupported (_deviceType) = true
    override _.Seed(seed) = Random.Seed(seed)
    override _.Zero(dtype, device) = SymbolicTensor(Shape.scalar, dtype, device) :> _
    override _.One(dtype, device) = SymbolicTensor(Shape.scalar, dtype, device) :> _
    override _.Zeros(shape:Shape, dtype, device) = SymbolicTensor(shape, dtype, device) :> _
    override _.Empty(shape:Shape, dtype, device) = SymbolicTensor(shape, dtype, device) :> _
    override _.Ones(shape:Shape, dtype, device) = SymbolicTensor(shape, dtype, device) :> _
    override _.Full(shape:Shape, _value:obj, dtype, device) = SymbolicTensor(shape, dtype, device) :> _
    override _.Random(shape:Shape, dtype, device) = SymbolicTensor(shape, dtype, device) :> _
    override _.RandomNormal(shape:Shape, dtype, device) = SymbolicTensor(shape, dtype, device) :> _
    override _.RandomInt(shape:Shape, _low:int, _high:int, dtype, device) = SymbolicTensor(shape, dtype, device) :> _
    override _.CreateFromFlatArray(_values:Array, shape, dtype, device) = SymbolicTensor(shape, dtype, device) :> _
