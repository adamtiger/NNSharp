using NNSharp.DataTypes;
using NNSharp.Kernels.CPUKernels;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static NNSharp.DataTypes.Data2D;
using static NNSharp.DataTypes.SequentialModelData;

namespace NNSharp.SequentialBased.SequentialLayers
{
    [Serializable()]
    public class MaxPool1DLayer : MaxPool1DKernel, ILayer
    {

        public MaxPool1DLayer(int padding, int stride, int kernelSize)
        {
            this.padding = padding;
            this.stride = stride;
            this.kernelDim.h = 1;
            this.kernelDim.w = kernelSize;
        }

        public IData GetOutput()
        {
            return output;
        }

        public void SetInput(IData input)
        {
            if (input == null)
                throw new Exception("MaxPool1DLayer: input is null.");
            else if (!(input is Data2D))
                throw new Exception("MaxPool1DLayer: input is not Data2D.");

            this.input = input as Data2D;

            Dimension dimI = this.input.GetDimension();

            kernelDim.c = 1;
            kernelDim.b = 1;

            int outputH = 1;
            int outputW = CalculateOutputSize1D(dimI.w, padding, stride, kernelDim.w);
            int outputC = dimI.c;
            int outputB = kernelDim.b;

            output = new Data2D(outputH, outputW, outputC, outputB);
        }

        public void SetWeights(IData weights)
        {
            // No weights.
        }

        public LayerData GetLayerSummary()
        {
            Dimension dimI = input.GetDimension();
            Dimension dimO = output.GetDimension();
            return new LayerData(
                this.ToString(),
                dimI.h, dimI.w, 1, dimI.c, dimI.b,
                dimO.h, dimO.w, 1, dimO.c, dimO.b);
        }


        private int CalculateOutputSize1D(int inpSize, int padding, int stride, int kernel)
        {
            return 1 + (inpSize + 2 * padding - kernel) / stride;
        }
    }
}
