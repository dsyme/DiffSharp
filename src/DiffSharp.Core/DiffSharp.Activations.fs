namespace DiffSharp

    [<AutoOpen>]
    module ActivationsExtensions =

        /// Defines an extension as an element-wise operation using the function and its derivative
        let inline elementwiseOp f deriv =
            Tensor.Op
                { new UnaryOp() with 
                    member _.ComputeRaw(a) = f a
                    member t.Forward(fab, a, ad) = t.Reverse(fab, a, ad)
                    member _.Reverse(t, a, td) = deriv t a td
                    }

        type Tensor with
            /// <summary>Applies the rectified linear unit function element-wise.</summary>
            member a.relu() = elementwiseOp (fun a -> a.ReluT()) (fun _t a td -> let sap = a.sign() in td * sap.abs() * (sap + 1.) / 2.) a

            /// <summary>Applies the leaky rectified linear unit function element-wise</summary>
            /// <remarks>\[\text{LeakyReLU}(x) = \max(0, x) + \text{negative\_slope} * \min(0, x)\]</remarks>
            /// <param name="negativeSlope">Controls the angle of the negative slope. Default: 0.01.</param>
            member a.leakyRelu(?negativeSlope:float) =
                let negativeSlope = defaultArg negativeSlope 0.01
                let zeros = a.zerosLike() in zeros.max(a) + negativeSlope * zeros.min(a)

            /// <summary>Applies the rectified linear unit function element-wise.</summary>
            member a.elu() = elementwiseOp (fun a -> a.EluT()) (fun t a td -> failwith "deriv of elu NYI") a

            /// <summary>Applies the silu function element-wise.</summary>
            member a.silu() = elementwiseOp (fun a -> a.SiluT()) (fun t a td -> failwith "deriv of silu NYI") a

            /// <summary>Applies the gelu function element-wise.</summary>
            member a.gelu() = elementwiseOp (fun a -> a.GeluT()) (fun t a td -> failwith "deriv of gelu NYI") a

            /// <summary>Applies the hardswish function element-wise.</summary>
            member a.hardswish() = elementwiseOp (fun a -> a.HardswishT()) (fun t a td -> failwith "deriv of hardswish NYI") a

            /// <summary>Applies the relu6 function element-wise.</summary>
            member a.relu6() = elementwiseOp (fun a -> a.Relu6T()) (fun t a td -> failwith "deriv of relu6 NYI") a

            /// <summary>Applies the relu6 function element-wise.</summary>
            member a.hardsigmoid() = elementwiseOp (fun a -> a.HardsigmoidT()) (fun t a td -> failwith "deriv of hardsigmoid NYI") a

            /// <summary>Applies the sigmoid function element-wise.</summary>
            member a.sigmoid() = elementwiseOp (fun a -> a.SigmoidT()) (fun t _a td -> td * t * (1. - t)) a

            /// <summary>Applies the softplus function element-wise.</summary>
            /// <remarks>\[\text{Softplus}(x) = \frac{1}{\beta} * \log(1 + \exp(\beta * x))\]</remarks>
            /// <summary>Applies the relu6 function element-wise.</summary>
            member a.softplus() = elementwiseOp (fun a -> a.SoftplusT()) (fun _t a td -> td / (1. + a.neg().exp())) a

        type dsharp with

            /// <summary>Applies the rectified linear unit function element-wise.</summary>
            /// <param name="input">The input tensor.</param>
            static member relu(input:Tensor) = input.relu()

            /// <summary>Applies the leaky rectified linear unit function element-wise</summary>
            /// <remarks>\[\text{LeakyReLU}(x) = \max(0, x) + \text{negative\_slope} * \min(0, x)\]</remarks>
            /// <param name="input">The input tensor.</param>
            /// <param name="negativeSlope">Controls the angle of the negative slope. Default: 0.01.</param>
            static member leakyRelu(input:Tensor, ?negativeSlope:float) = input.leakyRelu(?negativeSlope=negativeSlope)

            /// <summary>Applies the elu function element-wise.</summary>
            /// <param name="input">The input tensor.</param>
            static member elu(input:Tensor) = input.elu()

            /// <summary>Applies the gelu function element-wise.</summary>
            /// <param name="input">The input tensor.</param>
            static member silu(input:Tensor) = input.silu()

            /// <summary>Applies the gelu function element-wise.</summary>
            /// <param name="input">The input tensor.</param>
            static member gelu(input:Tensor) = input.gelu()

            /// <summary>Applies the hardswish function element-wise.</summary>
            /// <param name="input">The input tensor.</param>
            static member hardswish(input:Tensor) = input.hardswish()

            /// <summary>Applies the hardswish function element-wise.</summary>
            /// <param name="input">The input tensor.</param>
            static member relu6(input:Tensor) = input.relu6()

            /// <summary>Applies the hardsigmoid function element-wise.</summary>
            /// <param name="input">The input tensor.</param>
            static member hardsigmoid(input:Tensor) = input.hardsigmoid()

            /// <summary>Applies the sigmoid element-wise function</summary>
            /// <remarks>\[\text{Sigmoid}(x) = \frac{1}{1 + \exp(-x)}\]</remarks>
            /// <param name="input">The input tensor.</param>
            static member sigmoid(input:Tensor) = input.sigmoid()

            /// <summary>Applies the softplus function element-wise.</summary>
            /// <remarks>\[\text{Softplus}(x) = \frac{1}{\beta} * \log(1 + \exp(\beta * x))\]</remarks>
            /// <param name="input">The input tensor.</param>
            static member softplus(input:Tensor) = input.softplus()

namespace DiffSharp.Compose

    open DiffSharp

    [<AutoOpen>]
    module ActivationsExtensions =

        type dsharp with
            /// <summary>TBD</summary>
            static member leakyRelu(negativeSlope:float) = fun (a:Tensor) -> a.leakyRelu(negativeSlope=negativeSlope)

