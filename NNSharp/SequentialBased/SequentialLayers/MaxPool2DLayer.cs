using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NNSharp.DataTypes;
using NNSharp.Kernels.CPUKernels;
using static NNSharp.DataTypes.Data2D;

namespace NNSharp.SequentialBased.SequentialLayers
{
    public class MaxPool2DLayer : MaxPool2DKernel, ILayer
    {

        public MaxPool2DLayer(int paddingVertical, int paddingHorizontal,
                              int strideVertical, int strideHorizontal, 
                              int kernelHeight, int kernelWidth)
        {
            this.paddingVertical = paddingVertical;
            this.paddingHorizontal = paddingHorizontal;
            this.strideVertical = strideVertical;
            this.strideHorizontal = strideHorizontal;
            this.kernelDim.h = kernelHeight;
            this.kernelDim.w = kernelWidth;
        }

        public IData GetOutput()
        {
            return output;
        }

        public void SetInput(IData input)
        {
            if (input == null)
                throw new Exception("MaxPool2DLayer: input is null.");
            else if (!(input is Data2D))
                throw new Exception("MaxPool2DLayer: input is not Data2D.");

            this.input = input as Data2D;

            Dimension dimI = this.input.GetDimension();

            kernelDim.c = 1;
            kernelDim.b = 1;

            int outputH = CalculateOutputSize1D(dimI.h, paddingVertical, strideVertical, kernelDim.h);
            int outputW = CalculateOutputSize1D(dimI.w, paddingHorizontal, strideHorizontal, kernelDim.w);
            int outputC = dimI.c;
            int outputB = kernelDim.b;

            output = new Data2D(outputH, outputW, outputC, outputB);
        }

        public void SetWeights(IData weights)
        {
            // No weights.
        }

        private int CalculateOutputSize1D(int inpSize, int padding, int stride, int kernel)
        {
            return 1 + (inpSize + 2 * padding - kernel) / stride;
        }
    }
}
