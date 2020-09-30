namespace DiffSharp.Tools.Shapes

open System
open System.Collections.Generic

type InferenceVarSoln<'T> =
    | Solved of 'T
    | Unsolved
    
type InferenceVar<'T>(canSolve: bool) = 
    let mutable solution: InferenceVarSoln<'T> = Unsolved
    
    member __.IsSolved = match solution with Solved _ -> true | Unsolved -> false
    member __.CanSolve = canSolve

    member __.Solve sln = 
        if canSolve then 
            solution <- Solved sln
        else 
            failwith "can't solve"

    member __.Solution = solution

    member v.Id = string (hash v % 10345151) // reference hash reduced

/// Represents an inferred dimension
type TDim =
    internal
    /// One dimension is a multiple of another
    | DimMulInt of TDim * int

    /// One dimension is a divisor of another, striding semantics
    | DimDivInt of TDim * int

    /// The dimension is a variable, possibly solved
    | DimVar of InferenceVar<TDim>

    /// The dimension is named
    | DimNamed of string * TDim

    /// The dimension is known
    | DimKnown of int

    member dim.StripSolutions() = 
        match dim with 
        | DimNamed (name, dim2) -> 
            DimNamed (name, dim2.StripSolutions())
        | DimVar var -> 
            match var.Solution with 
            | Unsolved -> dim
            | Solved v -> v.StripSolutions()
        | _ -> dim

    /// Try to get the solved value for the dimension.
    // Note this is lossy on the name
    member dim.TryValue(ndeep) = 
        let rec loop ndeep (dim: TDim) = 
            if ndeep > 20 then None else
            match dim.StripSolutions() with 
            | DimNamed (_name, dim2) -> loop (ndeep+1) dim2 //.TryValue()
            | DimMulInt (expected,n) -> match loop (ndeep+1) expected (* .TryValue() *) with None -> None | Some dimv -> Some (dimv*n) 
            | DimDivInt (expected,n) -> match loop (ndeep+1) expected (* .TryValue() *) with None -> None | Some dimv -> Some (dimv/n + (if dimv % n > 0 then 1 else 0)) 
            | DimKnown n -> Some n 
            | DimVar _v -> None 
        loop ndeep dim

    member dim.FreeVarsAcc(acc: HashSet<_>) = 
        match dim.StripSolutions() with 
        | DimNamed (_name, dim2) -> dim2.FreeVarsAcc(acc)
        | DimMulInt (dim2,_n) -> dim2.FreeVarsAcc(acc)
        | DimDivInt (dim2,_n) -> dim2.FreeVarsAcc(acc)
        | DimKnown _n -> ()
        | DimVar v -> acc.Add(v)  |> ignore

    member dim.TryName() = 
        match dim.StripSolutions() with 
        | DimNamed (nm, dim2) -> Some (nm, dim2)
        | _ -> None

    member dim.HasName = dim.TryName().IsSome 

    member dim.IsSolved = dim.TryValue(0).IsSome
    
    member dim.ValueOrMinusOne = 
        match dim.TryValue(0) with 
        | Some value -> value
        | None -> -1

    member dim.ValueOrZero = 
        match dim.TryValue(0) with 
        | Some value -> value
        | None -> 0

    member dim.Value = 
        match dim.TryValue(0) with 
        | Some value -> value
        | None -> failwith "the value for the dimension could not be inferred"

    //member dim.AsTFNode(subst: IDictionary<InferenceVar<Dim>, (bool * Tensor)>) = 
    //    let rec loop (d: Dim) = 
    //        match d.StripSolutions() with 
    //        | DimMulInt (dim2, n) -> tf.multiply(loop dim2, tf.constant(n))
    //        | DimDivInt (dim2, n) -> tf.floordiv(gen_ops.add(loop dim2, tf.constant(n-1)), tf.constant(n))
    //        //| DimDivInt (dim2, n) -> tf.divide(tf.add(loop dim2, tf.constant(n-1)), tf.constant(n))
    //        | DimKnown n -> tf.constant(n)

    //        | DimNamed (_, dim2) -> loop dim2
    //        | DimVar v -> 
    //           if subst.ContainsKey(v) then 
    //               snd subst.[v] 
    //           else 
    //               //printfn "Dim.AsTFNode: didn'T find instantiation for variable dimension in %A, assuming 1" dim
    //               tf.constant(1)
    //    loop dim

    member dim.Subst(subst: IDictionary<InferenceVar<TDim>, TDim>) = 
        match dim.StripSolutions() with 
        | DimMulInt (dim2, n) -> DimMulInt (dim2.Subst(subst), n)
        | DimDivInt (dim2, n) -> DimDivInt (dim2.Subst(subst), n) 
        | DimKnown n -> DimKnown n
        | DimNamed (nm, dim2) -> DimNamed (nm, dim2.Subst(subst))
        | DimVar v -> 
            if subst.ContainsKey(v) then subst.[v] 
            else 
                //printfn "Dim.Subst: didn'T find instantiation for variable dimension in %A, assuming unchanged" dim
                dim

    static member ( * ) (dim: TDim, stride: int) = if stride = 1 then dim else DimMulInt (dim, stride)

    static member ( / ) (dim: TDim, stride: int) = if stride = 1 then dim else DimDivInt (dim, stride)

    static member Known value = DimKnown value

    /// A dimension with an inferred value
    static member Inferred = DimVar (InferenceVar(true))

    /// A named dimension with a known value
    static member Named name value = DimNamed (name, TDim.Known value)

    /// A dimension variable that gets inferred statically
    static member InferredVar name = DimNamed (name, DimVar (InferenceVar(true)))

    /// A dimension variable that is always variable and part of the input, for example
    static member Var name = DimNamed (name, DimVar (InferenceVar(false)))
    
    // A dimension variable that gets solved on the graph
    static member ExistentialVar name = let v = (InferenceVar(false)) in v, DimNamed (name, DimVar v)

    static member Unify op (actual: TDim) (expected: TDim) = 
        match TDim.UnifyInner op actual expected with
        | Ok () -> ()
        | Error msg -> failwithf "mismatched dimensions for operator %s: expected '%s' but got '%s' (%s)" op (expected.ToString())  (actual.ToString()) msg

    static member private occurs (v: InferenceVar<TDim>) (soln: TDim) = 
        match soln.StripSolutions() with 
        | DimVar v2 when Object.ReferenceEquals(v,v2) -> true
        | DimMulInt (d, _) -> TDim.occurs v d
        | DimDivInt (d, _) -> TDim.occurs v d
        | DimNamed (_, d) -> TDim.occurs v d
        | DimVar _ -> false
        | DimKnown _ -> false

    static member private solve (v: InferenceVar<TDim>) (vexp: TDim) (soln: TDim) = 
        if TDim.occurs v soln then 
            Error (sprintf "dimension expression '%s = %s' would be infinite" (vexp.ToString()) (soln.ToString()))
        else
            v.Solve soln
            Ok()

    static member UnifyInner op (actual: TDim) (expected: TDim) = 
        //use _holder = enter "Dim - UnifyInner"
        match actual.TryValue(0), expected.TryValue(0) with 
        | Some v1, Some v2 -> if v1 <> v2 then Error "unequal values" else Ok()
        | _ -> 
        match actual.StripSolutions(), expected.StripSolutions() with 
        // check for identical variables
        | DimVar var1, DimVar var2 when Object.ReferenceEquals(var1,var2) -> Ok ()
        // solve
        | DimVar var1, _ when var1.CanSolve -> 
            TDim.solve var1 actual expected

        | _, DimVar var2 when var2.CanSolve -> 
            TDim.solve var2 expected actual

        | DimKnown _d1, DimKnown _d2 -> failwith "unreachable - each dimension had value"

        | DimMulInt (d1, n1), DimKnown d2 -> 
            if d2 % n1 <> 0 then 
                Error "not divisible"
            else
                TDim.UnifyInner op d1 (DimKnown (d2 / n1))
        | DimKnown d1, DimMulInt (d2, n2) -> 
            if d1 % n2 <> 0 then 
                Error "not divisible"
            else
                TDim.UnifyInner op (DimKnown (d1 / n2)) d2
        | DimMulInt (d1, n1), DimMulInt (d2, n2) -> 
            if n1 <> n2 then 
                Error "different multipliers"
            else
                TDim.UnifyInner op d1 d2
        | DimDivInt (d1, n1), DimDivInt (d2, n2) -> 
            if n1 <> n2 then 
                Error "different multipliers"
            else
                TDim.UnifyInner op d1 d2
        | DimNamed (name1, d1), DimNamed(name2, d2) when name1 = name2 -> 
            //if name1 <> name2 then 
            //    printfn "named dimension '%s' was equated with named dimension '%s'" name1 name2
            TDim.UnifyInner op d1 d2
        | DimNamed (_, d1), d2 
        | d1, DimNamed (_, d2) -> 
            TDim.UnifyInner op d1 d2
        | _ -> 
            match actual.TryValue(0), expected.TryValue(0) with 
            | None, _ | _, None -> Error "incomplete dimension"
            | _ -> Ok () // equal, see above

    override dim.ToString() = 
        let rec loop ndeep (dim: TDim) = 
            if ndeep > 20 then "" else 
            match dim.TryName() with 
            | Some (name, dim2) -> 
                let slntext = loop (ndeep+1) dim2 // .ToString()
                name + (if slntext = "?" then "" else " (=" + slntext + ")")
            | None -> 
            // Check if it is a computed constant
            // We currently prefer this to showing symbolic names, e.g. N/2 or N/2*2
            match dim.TryValue(ndeep) with 
            | Some v -> string v
            | None ->  
            match dim.StripSolutions() with 
            | DimMulInt (expected, n) -> loop (ndeep+1) expected (* .ToString()  *) + "*" + string n 
            | DimDivInt (expected, n) -> loop (ndeep+1) expected (* .ToString() *) + "/" + string n 
            | DimKnown n -> string n 
            | DimNamed _ -> failwith "unreachable" 
            | DimVar v -> "?" + v.Id
        loop 0 dim

/// Represents an inferred shape
type TShape (flex: InferenceVar<TShape> option, suffix: TDim[]) =

    static let empty = TShape (None, [| |])

    member __.DimensionsWithFlexVar = 
        match flex with 
        | None -> None, suffix
        | Some v -> 
            match v.Solution with 
            | Unsolved -> Some v, suffix
            | Solved sln -> let flex, dims2 = sln.DimensionsWithFlexVar in flex, Array.append dims2 suffix

    member shape.DimensionsEliminatingFlex = 
        let flexvar, dims = shape.DimensionsWithFlexVar
        match flexvar with 
        | Some var when var.CanSolve -> var.Solve empty
        | _ -> ()
        dims 

    member shape.FreeVarsAcc(acc: HashSet<_>) = 
        let _flexvar, dims = shape.DimensionsWithFlexVar
        for dim in dims do dim.FreeVarsAcc(acc)

    static member FreeVars(shapes: TShape[]) = 
        let acc = HashSet(HashIdentity.Reference)
        for shape in shapes do shape.FreeVarsAcc(acc)
        acc |> Seq.toArray

    /// Get the final inferred rank 
    member shape.Rank = shape.DimensionsEliminatingFlex.Length

    /// Lookup an inferred dimension
    member shape.Item 
        with get idx = 
            let dims = shape.DimensionsEliminatingFlex 
            if idx < dims.Length then 
                dims.[idx]
            else
                failwithf "tensor has insufficient dimensions, at least %d required but only %d available" idx dims.Length

    /// Get the shape as a Rank-1 shape
    member shape.AsRank1() = 
        let dims = shape.DimensionsEliminatingFlex 
        if dims.Length <> 1 then invalidArg "AsRank1" "not a rank 1 shape"
        dims.[0]

    /// Get the shape as a Rank-2 shape
    member shape.AsRank2() =
        let dims = shape.DimensionsEliminatingFlex 
        if dims.Length <> 2 then invalidArg "AsRank2" "not a rank 2 shape"
        dims.[0], dims.[1]

    /// Get the shape as a Rank-3 shape
    member shape.AsRank3() = 
        let dims = shape.DimensionsEliminatingFlex 
        if dims.Length <> 3 then invalidArg "AsRank3" "not a rank 3 shape"
        dims.[0], dims.[1], dims.[2]

    /// Get the shape as a Rank-4 shape
    member shape.AsRank4() = 
        let dims = shape.DimensionsEliminatingFlex 
        if dims.Length <> 4 then invalidArg "AsRank4" "not a rank 4 shape"
        dims.[0], dims.[1], dims.[2], dims.[3]

    member shape.AsFlexRank3() = 
        let flex, dims = shape.DimensionsWithFlexVar
        let dimsBefore, dims = dims |> Array.splitAt (dims.Length - 3)
        flex, dimsBefore, dims.[dims.Length-3], dims.[dims.Length-2], dims.[dims.Length-1]

    member shape.AsRank3OrMore() = 
        let _flex, dimsBefore, a, b, c = shape.AsFlexRank3()
        dimsBefore, a, b, c

    /// Get the dimensions of the shape
    member shape.Dimensions = shape.DimensionsEliminatingFlex

    ///// Get the shape as a TensorFlow shape
    //member shape.AsTFShape() = 
    //    TensorShape(shape.DimensionsEliminatingFlex |> Array.map (fun dim -> int dim.ValueOrMinusOne))

    /// Get the shape as a TensorFlow node
    member shape.IsSolved = 
        shape.DimensionsEliminatingFlex |> Array.forall (fun dim -> dim.IsSolved)

    /// Get the shape as a TensorFlow node
    member shape.Subst(subst: IDictionary<InferenceVar<TDim>, TDim>) = 
        let dims = shape.DimensionsEliminatingFlex
        TShape.NoFlex [| for dim in dims -> dim.Subst(subst) |]

    ///// Get the shape as a TensorFlow node
    //member shape.AsTFNode(subst: IDictionary<InferenceVar<Dim>, (bool * Tensor)>) = 
    //    if shape.IsSolved then 
    //        tf.constant(shape.AsTFShape().dims) // NOTE: we're assuming tf.constant can handle shapes
    //    else
    //        let dims = shape.DimensionsEliminatingFlex
    //        if dims.Length = 0 then 
    //            tf.constant(shape.AsTFShape().dims) // NOTE: we're assuming tf.constant can handle shapes
    //        else
    //            let dimExprs = [| for dim in dims -> dim.AsTFNode(subst) |]
    //            gen_ops.pack dimExprs

    /// Copy the shape returning a map of old variables to new.  The new shape is then unified against another shape, giving a solution for those variables
    member shape.Match(shape2) = 
        //use _holder = enter "TShape - Match"
        let acc = Dictionary()
        let dims = shape.DimensionsEliminatingFlex
        let rec freshen (dim: TDim) = 
            match dim.StripSolutions() with 
            | DimMulInt (dim2, n) -> DimMulInt (freshen dim2, n) 
            | DimDivInt (dim2, n) -> DimDivInt (freshen dim2, n)
            | DimKnown _ -> dim
            | DimNamed (nm, dim2) -> DimNamed (nm, freshen dim2)
            | DimVar var -> 
                if acc.ContainsKey(var) then 
                    DimVar acc.[var] 
                else 
                    let newVar = InferenceVar(true)
                    acc.[var] <- newVar
                    DimVar newVar

        let dimsCopied = dims |> Array.map freshen 
        let shapeCopied = TShape.NoFlex dimsCopied
        //printfn "matching, shapeCopied = %A, shape2 = %A, shape = %A, #acc = %d" shapeCopied shape2 shape acc.Count
        TShape.Unify "match" shapeCopied shape2
        //printfn "after matching, shapeCopied = %A, shape2 = %A, shape = %A, #acc = %d" shapeCopied shape2 shape acc.Count
        [| for (KeyValue(v1, v2)) in acc do
                    match v2.Solution with 
                    | Unsolved -> failwith "the input shape didn'T solve a shape variable"
                    | Solved d -> yield (v1, d) |]

    /// Create a shape with the given dimension information. The shape does not support broadcasting.
    static member NoFlex dims = TShape(None, dims)

    /// Create a shape with the given dimension information. The shape does not support broadcasting.
    new (dims) = TShape(None, Array.ofSeq dims)
    new (dims:int[]) = TShape(None, Array.ofSeq (dims |> Array.map (function | -1 -> TDim.Inferred | n -> TDim.Known(n))))

    /// Create a shape with the given dimension information. The shape supports broadcasting to further initial dimensions.
    static member Flex dims = TShape(Some (InferenceVar(true)), dims)

    static member PossibleFlex (flex: bool) dims = if flex then TShape.Flex dims else TShape.NoFlex dims

    /// Create a new fully inferred shape 
    static member Inferred with get() = TShape.Flex [| |]
    
    /// Create a shape with the given dimensions. Same as 'shape [...]'.
    static member Known (ints: seq<int>) = 
        ints 
        |> Array.ofSeq 
        |> Array.map (fun i -> if i = -1 then TDim.Inferred else DimKnown i)
        |> TShape.NoFlex 

    /// Create a shape from a TensorFlow array of int64 values
    static member FromShape (shape: int[], ?flex: bool) = 
        let flex = defaultArg flex false
        let dims = shape |> Array.map (fun i -> if i = -1 then TDim.Inferred else DimKnown (int32 i))
        TShape.PossibleFlex flex dims

    ///// Create a shape from a TensorFlow shape
    //static member FromTFShape (shape: TensorShape) = 
    //    shape.dims |> TShape.FromShape

    static member Scalar = TShape Array.empty<TDim>
    
    static member Vector = TShape [| TDim.Inferred |]
    
    static member Matrix = TShape [| TDim.Inferred; TDim.Inferred |]

    /// At least 'n' dimensions, possible more
    static member FlexN n = TShape.Flex [| for _i in 1 .. n -> TDim.Inferred |]

    static member MinDimensions op (shape: TShape) dim = 
        let flexvar, dims = shape.DimensionsWithFlexVar
        if dim > dims.Length then 
            match flexvar with 
            | Some v when v.CanSolve -> 
                v.Solve (TShape.FlexN (dim - dims.Length))
            | _ -> 
                failwithf "shape %A must have at least %d dimensions for operator %s" shape dim op

    static member  Unify op (actual: TShape) (expected: TShape) = 

        let rec loop (s1: TShape) (s2: TShape) =
            //use _holder = enter "TShape - Unify - loop"

            let flexvar1, dims1 = s1.DimensionsWithFlexVar
            let flexvar2, dims2 = s2.DimensionsWithFlexVar

            // How many dimensions in common?
            let n = min dims1.Length dims2.Length
            let dims1A, dims1B = dims1 |> Array.splitAt (dims1.Length-n)
            let dims2A, dims2B = dims2 |> Array.splitAt (dims2.Length-n)
            
            // Unify the prefix
            let prefixRes = 
                if n > 0 then
                    // Drop front dimensions - shapes smaller
                    loop (TShape(flexvar1, dims1A)) (TShape(flexvar2, dims2A))

                elif dims1.Length > 0 then
                    assert (dims2.Length = 0)
                    match flexvar2 with 
                    | Some v2 when v2.CanSolve -> 
                        v2.Solve (TShape.FlexN dims1.Length) 
                        // expected now expanded and will have 'n' in common
                        loop s1 s2 
                    | _ -> 
                        Error ()

                elif dims2.Length > 0 then
                    assert (dims1.Length = 0)
                    match flexvar1 with 
                    | Some v1 when v1.CanSolve -> 
                        v1.Solve (TShape.FlexN dims2.Length) 
                        // actual now expanded and will have 'n' in common
                        loop s1 s2 
                    | _ -> 
                        Error ()

                else

                    match flexvar1, flexvar2 with 
                    | Some v1, Some v2 when Object.ReferenceEquals(v1,v2) -> Ok ()
                    | Some v1, _ when v1.CanSolve -> v1.Solve (TShape(flexvar2, [| |])); Ok()
                    | _, Some v2 when v2.CanSolve -> v2.Solve (TShape(flexvar1, [| |])); Ok()
                    | None, None -> Ok()
                    | _ -> Error ()
            match prefixRes with 

            | Error () -> Error()
            | Ok() -> 
                // Unify the common sufix
                (dims1B, dims2B) ||> Array.iter2 (fun dim1 dim2 ->
                    match TDim.UnifyInner op dim1 dim2 with 
                    | Ok () -> ()
                    | Error msg -> failwithf "mismatched shapes for operator %s: expected %A but got %A (size %s did not match %s, %s) " op expected actual (dim2.ToString()) (dim1.ToString()) msg)
                Ok()

        match loop actual expected with 
        | Ok () -> ()
        | Error () -> failwithf "mismatched shapes: expected %A but got %A for operator %s" expected actual op

    static member EquivShapes op (actual: TShape) (expected: TShape) = 
        TShape.Unify op actual expected
        // Preserve names from either
        let v1, dims1 = actual.DimensionsWithFlexVar
        let _v2, dims2 = expected.DimensionsWithFlexVar 
        let dims = (dims1, dims2) ||> Array.map2 (fun dim1 dim2 -> if dim2.HasName then dim2 else dim1)
        TShape(v1, dims)

    /// Convert the shape to a string
    override shape.ToString() = 
        let flexvar, dims = shape.DimensionsWithFlexVar
        if dims.Length = 0 && flexvar.IsSome then 
            "?" + flexvar.Value.Id
        elif dims.Length = 0 then 
            "scalar" 
            + (if flexvar.IsSome then " (can broadcast)" else "")
        elif dims.Length = 1 then 
            "vector " + dims.[0].ToString()
            + (if flexvar.IsSome then " (can broadcast)" else "")
        elif dims.Length = 2 then 
            "matrix " + dims.[0].ToString() + " x " + dims.[1].ToString()
            + (if flexvar.IsSome then " (can broadcast)" else "")
        else
            sprintf "shape %s" (String.concat " x " [ for i in dims -> i.ToString() ]) 
            + (if flexvar.IsSome then "x.." else "")
