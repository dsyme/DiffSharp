module DiffSharp.Symbols

open System
open System.Collections.Concurrent
open System.Reflection

let (|HasMethod|_|) (nm: string) (t: Type) =
    match t with 
    | null -> None
    | _ -> 
    match t.GetMethod(nm, BindingFlags.Static ||| BindingFlags.Public ||| BindingFlags.NonPublic) with
    | null -> None
    | m -> Some m

[<Struct>]
type symbol(syms: SymAllocator, code: int, name: string) =
    member _.SymAllocator = syms
    member _.Id = code   
    member _.Name = name
    override _.ToString() = "?" + name
    //member sym.Solution = syms.Solve(sym, v)
    member sym.Solve(v:obj) = syms.Solve(sym, v)

and SymAllocator() =
    
    let mutable last = 777000000
    let mapping = ConcurrentDictionary<int, string>()
    let solutions = ConcurrentDictionary<int, obj>()
    let constraints = ConcurrentQueue()
    member syms.CreateSymbol (name: string) : 'T =
        let code = last
        let symbol = symbol(syms, code, name)
        last <- last + 1
        mapping.[code] <- name
        let t = typeof<'T>

        match typeof<'T> with
        | t when t = typeof<int> -> code |> box |> unbox
        | HasMethod "Symbolic" m ->
            m.Invoke(null, [| box symbol |]) |> unbox
        | _ -> 
        match t.Assembly.GetType(t.FullName + "Module") with
        | HasMethod "Symbolic" m ->
             m.Invoke(null, [| box symbol |]) |> unbox
        | _ ->
            failwithf "no static 'Symbolic' method found in type '%s' or a partner module of the same name" t.FullName

    static member op_Dynamic (x: SymAllocator, name: string) = x.CreateSymbol(name)
    member _.Constrain(cx) = constraints.Enqueue(cx)
    member _.Solve(sym: symbol, v: obj) = solutions.[sym.Id] <- v

let sym : SymAllocator = SymAllocator()

//let v : int = (sym?A) 

