namespace DiffSharp

[<AutoOpen>]
module AvgPoolExtensions =

    type Tensor with
        /// <summary>Applies a 1D average pooling over an input signal composed of several input planes, returning the max indices along with the outputs.</summary>
        /// <param name="kernelSize">The size of the window to take a max over.</param>
        /// <param name="stride">The stride of the window. Default value is kernelSize.</param>
        /// <param name="padding">The implicit zero padding to be added on both sides.</param>
        member a.avgpool1d(kernelSize:int, ?stride:int, ?padding:int(* , ?ceil_mode: bool, ?count_include_pad: bool *)) =
            let stride = defaultArg stride kernelSize
            let padding = defaultArg padding 0
            //let ceil_mode = defaultArg ceil_mode false
            //let count_include_pad= defaultArg count_include_pad true
            Shape.checkCanAvgpool1d a.dtype a.shape kernelSize stride padding |> ignore
            Tensor.Op
                { new UnaryOp() with 
                    member _.ComputeRaw(a) = a.AvgPool1D(kernelSize, stride, padding(* , ceil_mode, count_include_pad *))
                    member t.Forward(fab, a, ad) = ad.avgpool1d(kernelSize, stride, padding(* , ceil_mode, count_include_pad *))
                    member _.Reverse(fab, a, td) = td.avgpoolReverse1d(a, kernelSize, stride, padding(* , ceil_mode, count_include_pad *))
                 }
                 a

        /// <summary>Computes a partial inverse of avgpool1d</summary>
        /// <param name="originalInput">The original input to avgpool1d, used for size information.</param>
        /// <param name="kernelSize">The size of the window to take a max over.</param>
        /// <param name="stride">The stride of the window. Default value is kernelSize.</param>
        /// <param name="padding">The implicit zero padding to be added on both sides.</param>
        member a.avgpoolReverse1d(originalInput:Tensor, kernelSize:int, ?stride:int, ?padding:int(* , ?ceil_mode: bool, ?count_include_pad: bool *)) =
            let stride = defaultArg stride kernelSize
            let padding = defaultArg padding 0
            //let ceil_mode = defaultArg ceil_mode false
            //let count_include_pad= defaultArg count_include_pad true
            Tensor.Op
                { new UnaryOp() with 
                    member _.ComputeRaw(a) = a.AvgPoolReverse1D(originalInput.primalRaw, kernelSize, stride, padding(* , ceil_mode, count_include_pad *))
                    member _.Forward(fab, a, ad) = ad.avgpoolReverse1d(originalInput, kernelSize, stride, padding(* , ceil_mode, count_include_pad *))
                    member _.Reverse(fab, a, td) = td.avgpool1d(kernelSize, stride, padding(* , ceil_mode, count_include_pad *))
                 }
                 a

        /// <summary>Applies a 1D average pooling over an input signal composed of several input planes, returning the max indices along with the outputs.</summary>
        /// <param name="kernelSize">The size of the window to take a max over.</param>
        /// <param name="stride">The stride of the window. Default value is kernelSize.</param>
        /// <param name="padding">The implicit zero padding to be added on both sides.</param>
        /// <param name="kernelSizes">The sizes of the window to take a max over.</param>
        /// <param name="strides">The strides of the window. Default value is kernelSize.</param>
        /// <param name="paddings">The implicit zero paddings to be added on both sides.</param>
        member a.avgpool2d(?kernelSize:int, ?stride:int, ?padding:int, ?kernelSizes:seq<int>, ?strides:seq<int>, ?paddings:seq<int>(* , ?ceil_mode: bool, ?count_include_pad: bool *)) =
            let kernelSizes, strides, paddings = Shape.resolve2dMaxPoolSizes kernelSize kernelSizes stride strides padding paddings
            //let ceil_mode = defaultArg ceil_mode false
            //let count_include_pad= defaultArg count_include_pad true
            Shape.checkCanAvgpool2d a.dtype a.shape kernelSizes strides paddings  |> ignore
            Tensor.Op
                { new UnaryOp() with 
                    member _.ComputeRaw(a) = a.AvgPool2D(kernelSizes, strides, paddings(* , ceil_mode, count_include_pad *))
                    member _.Forward(fab, a, ad) = ad.avgpool2d(kernelSizes=kernelSizes, strides=strides, paddings=paddings(* , ceil_mode=ceil_mode, count_include_pad=count_include_pad *))
                    member _.Reverse(fab, a, td) = td.avgpoolReverse2d(a, kernelSizes=kernelSizes, strides=strides, paddings=paddings(* , ceil_mode=ceil_mode, count_include_pad=count_include_pad *))
                 }
                 a

        /// <summary>Computes a partial inverse of avgpool2d</summary>
        /// <param name="originalInput">The original input to avgpool1d, used for size information.</param>
        /// <param name="kernelSize">The size of the window to take a max over.</param>
        /// <param name="stride">The stride of the window. Default value is kernelSize.</param>
        /// <param name="padding">The implicit zero padding to be added on both sides.</param>
        /// <param name="kernelSizes">The sizes of the window to take a max over.</param>
        /// <param name="strides">The strides of the window. Default value is kernelSize.</param>
        /// <param name="paddings">The implicit zero paddings to be added on both sides.</param>
        member a.avgpoolReverse2d(originalInput:Tensor, ?kernelSize:int, ?stride:int, ?padding:int, ?kernelSizes:seq<int>, ?strides:seq<int>, ?paddings:seq<int>(* , ?ceil_mode: bool, ?count_include_pad: bool *)) =
            let kernelSizes, strides, paddings = Shape.resolve2dMaxPoolSizes kernelSize kernelSizes stride strides padding paddings
            //let ceil_mode = defaultArg ceil_mode false
            //let count_include_pad= defaultArg count_include_pad true
            Tensor.Op
                { new UnaryOp() with 
                    member _.ComputeRaw(a) = a.AvgPoolReverse2D(originalInput.primalRaw, kernelSizes, strides, paddings(* , ceil_mode, count_include_pad *))
                    member _.Forward(fab, a, ad) = ad.avgpoolReverse2d(originalInput, kernelSizes=kernelSizes, strides=strides, paddings=paddings(* , ceil_mode=ceil_mode, count_include_pad=count_include_pad *))
                    member _.Reverse(fab, a, td) = td.avgpool2d(kernelSizes=kernelSizes, strides=strides, paddings=paddings(* , ceil_mode=ceil_mode, count_include_pad=count_include_pad *))
                 }
                 a

        /// <summary>Applies a 3D average pooling over an input signal composed of several input planes, returning the max indices along with the outputs.</summary>
        /// <param name="kernelSize">The size of the window to take a max over.</param>
        /// <param name="stride">The stride of the window. Default value is kernelSize.</param>
        /// <param name="padding">The implicit zero padding to be added on both sides.</param>
        /// <param name="kernelSizes">The sizes of the window to take a max over.</param>
        /// <param name="strides">The strides of the window. Default value is kernelSize.</param>
        /// <param name="paddings">The implicit zero paddings to be added on both sides.</param>
        member a.avgpool3d(?kernelSize:int, ?stride:int, ?padding:int, ?kernelSizes:seq<int>, ?strides:seq<int>, ?paddings:seq<int>(* , ?ceil_mode: bool, ?count_include_pad: bool *)) =
            let kernelSizes, strides, paddings = Shape.resolve3dMaxPoolSizes kernelSize kernelSizes stride strides padding paddings
            //let ceil_mode = defaultArg ceil_mode false
            //let count_include_pad= defaultArg count_include_pad true
            Shape.checkCanAvgpool3d a.dtype a.shape kernelSizes strides paddings  |> ignore
            Tensor.Op
                { new UnaryOp() with 
                    member _.ComputeRaw(a) = a.AvgPool3D(kernelSizes, strides, paddings(* , ceil_mode, count_include_pad *))
                    member _.Forward(fab, a, ad) = ad.avgpool3d(kernelSizes=kernelSizes, strides=strides, paddings=paddings(* , ceil_mode=ceil_mode, count_include_pad=count_include_pad *))
                    member _.Reverse(fab, a, td) = td.avgpoolReverse3d(a, kernelSizes=kernelSizes, strides=strides, paddings=paddings(* , ceil_mode=ceil_mode, count_include_pad=count_include_pad *))
                 }
                 a

        /// <summary>Computes a partial inverse of avgpool1d</summary>
        /// <param name="originalInput">The original input to avgpool1d, used for size information.</param>
        /// <param name="kernelSize">The size of the window to take a max over.</param>
        /// <param name="stride">The stride of the window. Default value is kernelSize.</param>
        /// <param name="padding">The implicit zero padding to be added on both sides.</param>
        /// <param name="kernelSizes">The sizes of the window to take a max over.</param>
        /// <param name="strides">The strides of the window. Default value is kernelSize.</param>
        /// <param name="paddings">The implicit zero paddings to be added on both sides.</param>
        member a.avgpoolReverse3d(originalInput:Tensor, ?kernelSize:int, ?stride:int, ?padding:int, ?kernelSizes:seq<int>, ?strides:seq<int>, ?paddings:seq<int>(* , ?ceil_mode: bool, ?count_include_pad: bool *)) =
            let kernelSizes, strides, paddings = Shape.resolve3dMaxPoolSizes kernelSize kernelSizes stride strides padding paddings
            //let ceil_mode = defaultArg ceil_mode false
            //let count_include_pad= defaultArg count_include_pad true
            Tensor.Op
                { new UnaryOp() with 
                    member _.ComputeRaw(a) = a.AvgPoolReverse3D(originalInput.primalRaw, kernelSizes, strides, paddings(* , ceil_mode, count_include_pad *))
                    member _.Forward(fab, a, ad) = ad.avgpoolReverse3d(originalInput, kernelSizes=kernelSizes, strides=strides, paddings=paddings(* , ceil_mode=ceil_mode, count_include_pad=count_include_pad *))
                    member _.Reverse(fab, a, td) = td.avgpool3d(kernelSizes=kernelSizes, strides=strides, paddings=paddings(* , ceil_mode=ceil_mode, count_include_pad=count_include_pad *))
                 }
                 a

    type dsharp with
        /// <summary>Applies a 1D average pooling over an input signal composed of several input planes, returning the max indices along with the outputs.</summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="kernelSize">The size of the window to take a max over.</param>
        /// <param name="stride">The stride of the window. Default value is kernelSize.</param>
        /// <param name="padding">The implicit zero padding to be added on both sides.</param>
        static member avgpool1d(input: Tensor, kernelSize:int, ?stride:int, ?padding:int(* , ?ceil_mode: bool, ?count_include_pad: bool *)) =
            input.avgpool2d(kernelSize=kernelSize, ?stride=stride, ?padding=padding(* , ?ceil_mode=ceil_mode, ?count_include_pad=count_include_pad *))

        /// <summary>Computes a partial inverse of avgpool1d</summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="originalInput">The original input to avgpool1d, used for size information.</param>
        /// <param name="kernelSize">The size of the window to take a max over.</param>
        /// <param name="stride">The stride of the window. Default value is kernelSize.</param>
        /// <param name="padding">The implicit zero padding to be added on both sides.</param>
        static member avgpoolReverse1d(input: Tensor, originalInput:Tensor, kernelSize:int, ?stride:int, ?padding:int(* , ?ceil_mode: bool, ?count_include_pad: bool *)) =
            input.avgpoolReverse2d(originalInput, kernelSize=kernelSize, ?stride=stride, ?padding=padding(* , ?ceil_mode=ceil_mode, ?count_include_pad=count_include_pad *))

        /// <summary>Applies a 2D average pooling over an input signal composed of several input planes, returning the max indices along with the outputs.</summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="kernelSize">The size of the window to take a max over.</param>
        /// <param name="stride">The stride of the window. Default value is kernelSize.</param>
        /// <param name="padding">The implicit zero padding to be added on both sides.</param>
        /// <param name="kernelSizes">The sizes of the window to take a max over.</param>
        /// <param name="strides">The strides of the window. Default value is kernelSize.</param>
        /// <param name="paddings">The implicit zero paddings to be added on both sides.</param>
        static member avgpool2d(input: Tensor, ?kernelSize:int, ?stride:int, ?padding:int, ?kernelSizes:seq<int>, ?strides:seq<int>, ?paddings:seq<int>(* , ?ceil_mode: bool, ?count_include_pad: bool *)) =
            input.avgpool2d(?kernelSize=kernelSize, ?stride=stride, ?padding=padding, ?kernelSizes=kernelSizes, ?strides=strides, ?paddings=paddings(* , ?ceil_mode=ceil_mode, ?count_include_pad=count_include_pad *))

        /// <summary>Computes a partial inverse of avgpool2d</summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="originalInput">The original input to avgpool2d, used for size information.</param>
        /// <param name="kernelSize">The size of the window to take a max over.</param>
        /// <param name="stride">The stride of the window. Default value is kernelSize.</param>
        /// <param name="padding">The implicit zero padding to be added on both sides.</param>
        /// <param name="kernelSizes">The sizes of the window to take a max over.</param>
        /// <param name="strides">The strides of the window. Default value is kernelSize.</param>
        /// <param name="paddings">The implicit zero paddings to be added on both sides.</param>
        static member avgpoolReverse2d(input: Tensor, originalInput:Tensor, ?kernelSize:int, ?stride:int, ?padding:int, ?kernelSizes:seq<int>, ?strides:seq<int>, ?paddings:seq<int>(* , ?ceil_mode: bool, ?count_include_pad: bool *)) =
            input.avgpoolReverse2d(originalInput, ?kernelSize=kernelSize, ?stride=stride, ?padding=padding, ?kernelSizes=kernelSizes, ?strides=strides, ?paddings=paddings(* , ?ceil_mode=ceil_mode, ?count_include_pad=count_include_pad *))

        /// <summary>Applies a 2D average pooling over an input signal composed of several input planes, returning the max indices along with the outputs.</summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="kernelSize">The size of the window to take a max over.</param>
        /// <param name="stride">The stride of the window. Default value is kernelSize.</param>
        /// <param name="padding">The implicit zero padding to be added on both sides.</param>
        /// <param name="kernelSizes">The sizes of the window to take a max over.</param>
        /// <param name="strides">The strides of the window. Default value is kernelSize.</param>
        /// <param name="paddings">The implicit zero paddings to be added on both sides.</param>
        static member avgpool3d(input: Tensor, ?kernelSize:int, ?stride:int, ?padding:int, ?kernelSizes:seq<int>, ?strides:seq<int>, ?paddings:seq<int>(* , ?ceil_mode: bool, ?count_include_pad: bool *)) =
            input.avgpool3d(?kernelSize=kernelSize, ?stride=stride, ?padding=padding, ?kernelSizes=kernelSizes, ?strides=strides, ?paddings=paddings(* , ?ceil_mode=ceil_mode, ?count_include_pad=count_include_pad *))

        /// <summary>Computes a partial inverse of avgpool3d</summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="originalInput">The original input to avgpool3d, used for size information.</param>
        /// <param name="kernelSize">The size of the window to take a max over.</param>
        /// <param name="stride">The stride of the window. Default value is kernelSize.</param>
        /// <param name="padding">The implicit zero padding to be added on both sides.</param>
        /// <param name="kernelSizes">The sizes of the window to take a max over.</param>
        /// <param name="strides">The strides of the window. Default value is kernelSize.</param>
        /// <param name="paddings">The implicit zero paddings to be added on both sides.</param>
        static member avgpoolReverse3d(input: Tensor, originalInput:Tensor, ?kernelSize:int, ?stride:int, ?padding:int, ?kernelSizes:seq<int>, ?strides:seq<int>, ?paddings:seq<int>(* , ?ceil_mode: bool, ?count_include_pad: bool *)) =
            input.avgpoolReverse3d(originalInput, ?kernelSize=kernelSize, ?stride=stride, ?padding=padding, ?kernelSizes=kernelSizes, ?strides=strides, ?paddings=paddings(* , ?ceil_mode=ceil_mode, ?count_include_pad=count_include_pad *))

