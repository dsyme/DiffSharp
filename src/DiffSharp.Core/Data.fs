// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// and other contributors, see LICENSE in root of repository.
//
// BSD 2-Clause License. See LICENSE in root of repository.

namespace rec DiffSharp.Data

open DiffSharp
open DiffSharp.Compose
open DiffSharp.Util
open System.Net
open System.IO
open System.IO.Compression


/// <namespacedoc>
///   <summary>Contains datasets and components related to data loading.</summary>
/// </namespacedoc>
///
/// <summary>Represents a dataset.</summary>
[<AbstractClass>]
type Dataset() =
    abstract member length: int
    abstract member item: int -> Tensor * Tensor
    member d.loader(batchSize:int, ?shuffle:bool, ?dropLast:bool, ?dtype:Dtype, ?device:Device, ?backend:Backend, ?targetDtype:Dtype, ?targetDevice:Device, ?targetBackend:Backend) = DataLoader(d, batchSize=batchSize, ?shuffle=shuffle, ?dropLast=dropLast, ?dtype=dtype, ?device=device, ?backend=backend, ?targetDtype=targetDtype, ?targetDevice=targetDevice, ?targetBackend=targetBackend)
    member d.Item
        with get(i:int) =
            d.item(i)
    member d.GetSlice(imin:int option, imax:int option) =
        let imin   = defaultArg imin 0
        let imax   = defaultArg imax d.length
        if imin >= imax then failwithf "Expecting imin (%A) < imax (%A)" imin imax
        DatasetSubset(d, [|imin..imax|])
    member d.filter(predicate:Tensor->Tensor->bool) =
        let indices = ResizeArray<int>()
        for i in 0..d.length-1 do
            let data, target = d.item(i)
            if predicate data target then
                indices.Add(i)
        if indices.Count = 0 then failwithf "Could not find any data items for which the predicate is true"
        DatasetSubset(d, indices.ToArray())


type DatasetSubset(dataset:Dataset, indices:int[]) =
    inherit Dataset()
    override d.length = indices.Length
    override d.item(i) = dataset.item(indices.[i])


type DataLoader(dataset:Dataset, batchSize:int, ?shuffle:bool, ?dropLast:bool, ?dtype:Dtype, ?device:Device, ?backend:Backend, ?targetDtype:Dtype, ?targetDevice:Device, ?targetBackend:Backend) =
    let batchSize = min batchSize dataset.length
    let shuffle = defaultArg shuffle false
    let dropLast = defaultArg dropLast true
    let dtype = defaultArg dtype Dtype.Default
    let device = defaultArg device Device.Default
    let backend = defaultArg backend Backend.Default
    let targetDtype = defaultArg targetDtype dtype
    let targetDevice = defaultArg targetDevice device
    let targetBackend = defaultArg targetBackend backend
    let datalength = if dropLast then batchSize*(dataset.length/batchSize) else dataset.length
    member d.length = ((float datalength)/(float batchSize)) |> ceil |> int
    member d.epoch() =
        let indexer = if shuffle then Random.shuffledIndices datalength else id
        let indices = Seq.init datalength id |> Seq.map indexer
        let batchIndices = indices |> Seq.chunkBySize batchSize
        let batches = batchIndices |> Seq.map (Array.map dataset.item >> Array.unzip)
        batches |> Seq.mapi (fun i (data, target) -> i, data |> dsharp.stack |> dsharp.move(dtype, device, backend), target |> dsharp.stack |> dsharp.move(targetDtype, targetDevice, targetBackend))
    member d.batch() = let _, data, target = d.epoch() |> Seq.head in data, target


type TensorDataset(data:Tensor, target:Tensor) =
    inherit Dataset()
    do if data.shape.[0] <> target.shape.[0] then failwith "Expecting data and target to have the same size in the first dimension"
    override d.length = data.shape.[0]
    override d.item(i) = data.[i], target.[i]


type ImageDataset(path:string, ?fileExtension:string, ?resize:int*int, ?transform:Tensor->Tensor, ?targetTransform:Tensor->Tensor) =
    inherit Dataset()
    let fileExtension = defaultArg fileExtension "png"
    let transform = defaultArg transform id
    let targetTransform = defaultArg targetTransform id
    let subdirs = Directory.GetDirectories(path) |> Array.sort
    let filesInSubdirs = [|for subdir in subdirs do
                            let files = Directory.GetFiles(subdir, "*."+fileExtension)
                            if files.Length > 0 then files|]
    let _classes = filesInSubdirs.Length
    let data = [|for i in 0.._classes-1 do
                    let files = filesInSubdirs.[i]
                    yield! Array.map (fun file -> file, i) files|]
    let _classNames = Array.map (fun (f:string[]) -> DirectoryInfo(f.[0]).Parent.Name) filesInSubdirs
    member d.classes = _classes
    member d.classNames = _classNames
    override d.length = data.Length
    override d.item(i) =
        let fileName, category = data.[i]
        transform (dsharp.loadImage(fileName, ?resize=resize)), targetTransform (dsharp.tensor(category))


type CIFAR10(path:string, ?url:string, ?train:bool, ?transform:Tensor->Tensor, ?targetTransform:Tensor->Tensor) =
    inherit Dataset()
    let path = Path.Combine(path, "cifar10") |> Path.GetFullPath
    let pathExtracted = Path.Combine(path, "cifar-10-batches-bin")
    let train = defaultArg train true
    let cifar10mean = dsharp.tensor([0.4914, 0.4822, 0.4465]).view([3;1;1])
    let cifar10stddev = dsharp.tensor([0.247, 0.243, 0.261]).view([3;1;1])
    let transform = defaultArg transform (fun t -> (t - cifar10mean) / cifar10stddev)
    let targetTransform = defaultArg targetTransform id
    let url = defaultArg url "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
    let file = Path.Combine(path, Path.GetFileName(url))

    let loadCIFAR10 fileName =
        let br = new BinaryReader(File.OpenRead(fileName))
        [|for _ in 1..10000 do
            let label = br.ReadByte() |> dsharp.tensor
            let image = br.ReadBytes(3*1024) |> Array.map float32 |> dsharp.tensor |> dsharp.view([3; 32; 32]) // Mapping bytes to float32 before tensor construction is crucial, otherwise we have an issue with confusing byte with int8 that is destructive
            image/255, label
        |] |> Array.unzip |> fun (i, l) -> dsharp.stack(i), dsharp.stack(l)

    let data, target =
        Directory.CreateDirectory(path) |> ignore
        if not (File.Exists(file)) then download url file
        if not (Directory.Exists(pathExtracted)) then extractTarGz file path
        let files = [|"data_batch_1.bin"; "data_batch_2.bin"; "data_batch_3.bin"; "data_batch_4.bin"; "data_batch_5.bin"; "test_batch.bin"|] |> Array.map (fun f -> Path.Combine(pathExtracted, f))
        if train then
            files.[..4] |> Array.map loadCIFAR10 |> Array.unzip |> fun (d, t) -> dsharp.cat(d), dsharp.cat(t)
        else
            loadCIFAR10 files.[5]

    let _classNames = File.ReadAllLines(Path.Combine(pathExtracted, "batches.meta.txt")) |> Array.take 10
    member d.classes = _classNames.Length
    member d.classNames = _classNames
    override d.length = data.shape.[0]
    override d.item(i) = transform data.[i], targetTransform target.[i]


type CIFAR100(path:string, ?url:string, ?train:bool, ?transform:Tensor->Tensor, ?targetTransform:Tensor->Tensor) =
    inherit Dataset()
    let path = Path.Combine(path, "cifar100") |> Path.GetFullPath
    let pathExtracted = Path.Combine(path, "cifar-100-binary")
    let train = defaultArg train true
    let cifar100mean = dsharp.tensor([0.5071, 0.4867, 0.4408]).view([3;1;1])
    let cifar100stddev = dsharp.tensor([0.2675, 0.2565, 0.2761]).view([3;1;1])
    let transform = defaultArg transform (fun t -> (t - cifar100mean) / cifar100stddev)
    let targetTransform = defaultArg targetTransform id
    let url = defaultArg url "https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz"
    let file = Path.Combine(path, Path.GetFileName(url))

    let loadCIFAR100 fileName n =
        let br = new BinaryReader(File.OpenRead(fileName))
        [|for _ in 1..n do
            let labelCoarse = br.ReadByte() |> dsharp.tensor
            let labelFine = br.ReadByte() |> dsharp.tensor
            let image = br.ReadBytes(3*1024) |> Array.map float32 |> dsharp.tensor |> dsharp.view([3; 32; 32]) // Mapping bytes to float32 before tensor construction is crucial, otherwise we have an issue with confusing byte with int8 that is destructive
            image/255, labelCoarse, labelFine
        |] |> Array.unzip3 |> fun (i, lc, lf) -> dsharp.stack(i), dsharp.stack(lc), dsharp.stack(lf)

    let data, _, targetFine =
        Directory.CreateDirectory(path) |> ignore
        if not (File.Exists(file)) then download url file
        if not (Directory.Exists(pathExtracted)) then extractTarGz file path
        if train then loadCIFAR100 (Path.Combine(pathExtracted, "train.bin")) 50000
        else loadCIFAR100 (Path.Combine(pathExtracted, "test.bin")) 10000

    let _classNamesCoarse = File.ReadAllLines(Path.Combine(pathExtracted, "coarse_label_names.txt")) |> Array.take 20
    let _classNamesFine = File.ReadAllLines(Path.Combine(pathExtracted, "fine_label_names.txt")) |> Array.take 100
    member d.classes = _classNamesFine.Length
    member d.classNames = _classNamesFine
    override d.length = data.shape.[0]
    override d.item(i) = transform data.[i], targetTransform targetFine.[i]    


type MNIST(path:string, ?urls:seq<string>, ?train:bool, ?transform:Tensor->Tensor, ?targetTransform:Tensor->Tensor) =
    inherit Dataset()
    let path = Path.Combine(path, "mnist") |> Path.GetFullPath
    let train = defaultArg train true
    let transform = defaultArg transform (fun t -> (t - 0.1307) / 0.3081)
    let targetTransform = defaultArg targetTransform id
    let urls = List.ofSeq <| defaultArg urls (Seq.ofList
                   ["http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz";
                    "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz";
                    "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz";
                    "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"])
    let files = [for url in urls do Path.Combine(path, Path.GetFileName(url))]

    let loadMNISTImages(filename:string) (n:option<int>) =
        let r = new BinaryReader(new GZipStream(File.OpenRead(filename), CompressionMode.Decompress))
        let magicnumber = r.ReadInt32() |> IPAddress.NetworkToHostOrder
        match magicnumber with
        | 2051 -> // Images
            let maxitems = r.ReadInt32() |> IPAddress.NetworkToHostOrder
            let rows = r.ReadInt32() |> IPAddress.NetworkToHostOrder
            let cols = r.ReadInt32() |> IPAddress.NetworkToHostOrder
            let n = defaultArg n maxitems
            r.ReadBytes(n * rows * cols)
            |> Array.map float32 // Mapping bytes to float32 before tensor construction is crucial, otherwise we have an issue with confusing byte with int8 that is destructive
            |> dsharp.tensor
            |> dsharp.view ([n; 1; 28; 28])
            |> fun t -> t / 255
        | _ -> failwith "Given file is not in the MNIST format."
    let loadMNISTLabels(filename:string) (n:option<int>) =
        let r = new BinaryReader(new GZipStream(File.OpenRead(filename), CompressionMode.Decompress))
        let magicnumber = r.ReadInt32() |> IPAddress.NetworkToHostOrder
        match magicnumber with
        | 2049 -> // Labels
            let maxitems = r.ReadInt32() |> IPAddress.NetworkToHostOrder
            let n = defaultArg n maxitems
            r.ReadBytes(n)
            |> Array.map int
            |> dsharp.tensor
            |> dsharp.view ([n])
        | _ -> failwith "Given file is not in the MNIST format."

    let data, target = 
        Directory.CreateDirectory(path) |> ignore
        if train then
            if not (File.Exists(files.[0])) then download urls.[0] files.[0]
            if not (File.Exists(files.[1])) then download urls.[1] files.[1]
            loadMNISTImages files.[0] None, loadMNISTLabels files.[1] None
        else
            if not (File.Exists(files.[2])) then download urls.[2] files.[2]
            if not (File.Exists(files.[3])) then download urls.[3] files.[3]
            loadMNISTImages files.[2] None, loadMNISTLabels files.[3] None

    member d.classes = 10
    member d.classNames = Array.init 10 id |> Array.map string
    override d.length = data.shape.[0]
    override d.item(i) = transform data.[i], targetTransform target.[i]