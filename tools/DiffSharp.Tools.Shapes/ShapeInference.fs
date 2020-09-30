module DiffSharp.Tools.Shapes.Inference

open DiffSharp
open System.Reflection
open DiffSharp.Util
open DiffSharp.Optim
open DiffSharp.Data
open DiffSharp.Model
open FSharp.Quotations
open FSharp.Quotations.Patterns
open FSharp.Quotations.DerivedPatterns

[<RequireQualifiedAccess>]
type TE = 
    | Tensor of TShape
    | Lambda of TV * TE
    | Obj of (string * TE) list
    | Unknown of Expr * TShape
    | Var of TV
    | Call of string * TE list * TShape

and TV = TV of string * TShape

and [<RequireQualifiedAccess>] TS = 
    | Tensor of TShape
    | Func of TShape * TS
    | Obj of (string * TS) list

[<AutoOpen>]
module QuotationUtils = 
    let rec (|List|_|) q =
        match q with 
        | NewUnionCase(uc, [arg1; List tail]) when uc.Name = "Cons" -> Some (arg1 :: tail)
        | NewUnionCase(uc, []) when uc.Name = "Empty" -> Some []
        | _ -> None

    let (|Getter|_|) (prop: PropertyInfo) =
        match prop.GetGetMethod true with
        | null -> None
        | v -> Some v

    let (|MacroReduction|_|) (p: Expr) =

        match p with
        | Applications(Lambdas(vs, body), args) 
            when vs.Length = args.Length 
                 && (vs, args) ||> List.forall2 (fun vs args -> vs.Length = args.Length) ->
            let tab = Map.ofSeq (List.concat (List.map2 List.zip vs args))
            let body = body.Substitute tab.TryFind
            Some body

        // Macro
        | PropertyGet(None, Getter(MethodWithReflectedDefinition body), []) ->
            Some body

        // Macro
        | Call(None, MethodWithReflectedDefinition(Lambdas(vs, body)), args) ->
            let tab =
                (vs, args)
                ||> List.map2 (fun vs arg -> 
                    match vs, arg with
                    | [v], arg -> [(v, arg)]
                    | vs, NewTuple args -> List.zip vs args 
                    | _ -> List.zip vs [arg])
                |> List.concat |> Map.ofSeq
            let body = body.Substitute tab.TryFind
            Some body

        // Macro - eliminate 'let'.
        //
        // Always eliminate these:
        //    - function definitions
        //
        // Always eliminate these, which are representations introduced by F# quotations:
        //    - let v1 = v2
        //    - let v1 = tupledArg.Item*
        //    - let copyOfStruct = ...

        | Let(v, e, body) when (match e with
                                | Lambda _ -> true
                                | Var _ -> true
                                | TupleGet(Var tv, _) when tv.Name = "tupledArg" -> true
                                | _ when v.Name = "copyOfStruct" && v.Type.IsValueType -> true
                                | _ -> false) ->
            let body = body.Substitute (fun v2 -> if v = v2 then Some e else None)
            Some body

        | _ -> None

let rec conv (env: Map<Var, TV>) (q: Expr) =
    match q with 

    | SpecificCall <@@ dsharp.tensor @@> (None, _, [ Coerce (arg, _) ; _dtype; _device; _backend]) -> 
        // TODO: here we can match a rich algebra of N-dimensional expressions siwth some known lengths
        match arg with 
        // 1D lists of known length
        | List args when arg.Type = typeof<double list> || arg.Type = typeof<int list> -> 
            TE.Tensor(TShape.Known [| args.Length |])

        // 2D lists of known length
        // TODO: improve the check so all are same length not just first
        | List ((List args1 :: _) as args) when arg.Type = typeof<double list list> || arg.Type = typeof<int list list> -> 
            TE.Tensor(TShape.Known [| args.Length; args1.Length |])

        | _ when arg.Type = typeof<double> -> TE.Tensor(TShape.Scalar)
        | _ when arg.Type = typeof<double list> -> TE.Tensor(TShape.Vector)
        | _ when arg.Type = typeof<double list list> -> TE.Tensor(TShape.Matrix)
        | _ -> failwithf "unknown tensor initialization expression %A or type %A" arg arg.Type

    | Call (obj, mi, args) when q.Type = typeof<Tensor> -> 
         TE.Call(mi.Name, ((Option.toList obj @ args) |> List.map (conv env)), TShape.Inferred)

    | Lambda(v, body) -> 
        let tv = TV (v.Name, TShape.Inferred)
        let env = env.Add(v, tv)
        TE.Lambda(tv, conv env body)

    | NewTuple(elems) -> 
        TE.Obj (elems |> List.mapi (fun i e -> ("Item"+string i, conv env e)))

    | Var(v) -> 
        TE.Var env.[v]

    | MacroReduction(e) -> conv env e
    //| _ when q.Type = typeof<Tensor> -> 
    //    TE.Unknown (q, TShape.Inferred)
    | _ -> failwithf "unknown input '%A'" q

let rec shapeof (te: TE) =
    match te with
    | TE.Tensor (s) -> TS.Tensor s
    | TE.Lambda (TV(_,s), te) -> TS.Func(s, shapeof te)
    | TE.Obj els -> TS.Obj (els |> List.map (fun (s, e) -> (s, shapeof e)))
    | TE.Unknown (q, s) -> TS.Tensor s
    | TE.Var (TV(_,s)) -> TS.Tensor s
    | TE.Call (_, _, s) -> TS.Tensor s

let tshapeof te = match shapeof te with TS.Tensor t -> t | _ -> failwithf "not a tensor %A" te

let rec constrain (te: TE) =
    match te with
    | TE.Tensor (ts) -> ()
    | TE.Lambda (tv, te) -> constrain te
    | TE.Obj els -> ()
    | TE.Unknown (q, s) -> ()
    | TE.Var v -> ()
    | TE.Call ((("op_Addition" | "op_Subtraction" | "op_Multiply" | "op_Division") as op), [a;b], s) -> 
         TShape.EquivShapes op (tshapeof a) (TShape.EquivShapes op (tshapeof b) s) |> ignore
    | TE.Call (op, _, s) -> failwithf "unknown op '%A'" op

[<AutoOpen; AbstractClass>]
type Shapes internal () =
    static member infer([<ReflectedDefinition>] q: Expr<'T>) = 
        let te = conv Map.empty q
        let ts = shapeof te
        printfn "pre shape: %A" ts
        constrain te
        printfn "post shape: %A" ts

    static member fails ([<ReflectedDefinition>] q: Expr<'T>) = 
        let te = conv Map.empty q
        let ts = shapeof te
        printfn "pre shape: %A" ts
        try 
           constrain te
        with e -> 
           printfn "expected: %s" e.Message

