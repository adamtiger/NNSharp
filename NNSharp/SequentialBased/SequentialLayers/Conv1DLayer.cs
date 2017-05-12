using NNSharp.DataTypes;
using NNSharp.Kernels.CPUKernels;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static NNSharp.DataTypes.Data2D;

namespace NNSharp.SequentialBased.SequentialLayers
{
    [Serializable()]
    public class Conv1DLayer : Conv1DKernel, ILayer
    {
        public Conv1DLayer(int padding, int stride)
        {
            this.weights = weights as Data2D;
            this.padding = padding;
            this.stride = stride;
        }

        public IData GetOutput()
        {
            return output;
        }

        public void SetInput(IData input)
        {
            if (input == null)
                throw new Exception("Conv1DLayer: input is null.");
            else if (!(input is Data2D))
                throw new Exception("Conv1DLayer: input is not Data2D.");

            this.input = input as Data2D;

            Dimension dimI = this.input.GetDimension();
            Dimension dimK = this.weights.GetDimension();

            if (dimI.c != dimK.c)
                throw new Exception("Wrong kernel and input sizes: sizes of channels should match." +
                   " Now: dimI: " + dimI.c + " != dimK: " + dimK.c);

            int outputH = 1;
            int outputW = CalculateOutputSize1D(dimI.w, padding, stride, dimK.w);
            int outputC = dimK.b;
            int outputB = dimI.b;

            output = new Data2D(outputH, outputW, outputC, outputB);
        }

        public void SetWeights(IData weights)
        {
            if (weights == null)
                throw new Exception("Conv1DLayer: weights is null.");
            else if (!(weights is Data2D))
                throw new Exception("Conv1DLayer: weights is not Data2D.");

            this.weights = weights as Data2D;
        }

        private int CalculateOutputSize1D(int inpSize, int padding, int stride, int kernel)
        {
            return 1 + (inpSize + 2 * padding - kernel) / stride;
        }
    }
}
