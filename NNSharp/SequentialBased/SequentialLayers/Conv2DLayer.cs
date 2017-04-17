using NNSharp.Kernels.CPUKernels;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NNSharp.DataTypes;
using static NNSharp.DataTypes.Data2D;

namespace NNSharp.SequentialBased.SequentialLayers
{
    public class Conv2DLayer : Conv2DKernel, ILayer
    {
        public Conv2DLayer(int paddingVertical, int paddingHorizontal, 
                           int strideVertical, int strideHorizontal)
        {
            this.weights = weights as Data2D;
            this.paddingVertical = paddingVertical;
            this.paddingHorizontal = paddingHorizontal;
            this.strideVertical = strideVertical;
            this.strideHorizontal = strideHorizontal;
        }

        public IData GetOutput()
        {
            return output;
        }

        public void SetInput(IData input)
        {
            if (input == null)
                throw new Exception("Conv2DLayer: input is null.");
            else if (!(input is Data2D))
                throw new Exception("Conv2DLayer: input is not Data2D.");

            this.input = input as Data2D;

            Dimension dimI = this.input.GetDimension();
            Dimension dimK = this.weights.GetDimension();

            if (dimI.c != dimK.c)
                throw new Exception("Wrong kernel and input sizes: sizes of channels should match." + 
                   " Now: dimI: " + dimI.c + " != dimK: " + dimK.c);

            int outputH = CalculateOutputSize1D(dimI.h, paddingVertical, strideVertical, dimK.h);
            int outputW = CalculateOutputSize1D(dimI.w, paddingHorizontal, strideHorizontal, dimK.w);
            int outputC = dimK.b;
            int outputB = dimI.b;

            output = new Data2D(outputH, outputW, outputC, outputB);
        }

        public void SetWeights(IData weights)
        {
            if (weights == null)
                throw new Exception("Conv2DLayer: weights is null.");
            else if (!(weights is Data2D))
                throw new Exception("Conv2DLayer: weights is not Data2D.");

            this.weights = weights as Data2D;
        }

        private int CalculateOutputSize1D(int inpSize, int padding, int stride, int kernel)
        {
            return 1 + (inpSize + 2 * padding - kernel) / stride;
        } 
    }
}
