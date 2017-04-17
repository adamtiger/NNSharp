using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NNSharp.DataTypes;
using NNSharp.Kernels.CPUKernels;

namespace NNSharp.SequentialBased.SequentialLayers
{
    public class Bias2DLayer : Bias2DKernel, ILayer
    {

        public Bias2DLayer()
        {
        }
 
        public IData GetOutput()
        {
            return input;
        }

        public void SetInput(IData input)
        {
            if (input == null)
                throw new Exception("Bias2DLayer: input is null.");
            else if (!(input is Data2D))
                throw new Exception("Bias2DLayer: input is not Data2D.");

            this.input = input as Data2D; // Set the input value.

            int a, b;
            if ((a = this.input.GetDimension().c) != (b =this.biases.GetLength()))
                throw new Exception("Bias2DLayer: the number of biases is not suitable -> "+ a + " != " + b);
        }

        public void SetWeights(IData weights)
        {
            if (weights == null)
                throw new Exception("Bias2DLayer: biases is null.");
            else if (!(weights is DataArray))
                throw new Exception("Bias2DLayer: biases is not DataArray.");

            this.biases = weights as DataArray;
        }
    }
}
