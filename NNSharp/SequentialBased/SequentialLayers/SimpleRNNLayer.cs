using NNSharp.Kernels.CPUKernels;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NNSharp.DataTypes;
using NNSharp.KernelDescriptors;
using static NNSharp.DataTypes.Data2D;
using static NNSharp.DataTypes.SequentialModelData;
using NNSharp.Kernels;

namespace NNSharp.SequentialBased.SequentialLayers
{
    [Serializable()]
    public class SimpleRNNLayer : SimpleRNNKernel, ILayer
    {
        public SimpleRNNLayer(int units, int inputDim, ActivationLambda lambda)
        {
            this.units = units;
            this.inputDim = inputDim;
            this.lambda = lambda;

            prevOutput = new Data2D(1, 1, units, 1);
            h = new Data2D(1, 1, units, 1);
        }

        public IData GetOutput()
        {
            return output;
        }

        public void SetInput(IData input)
        {
            if (input == null)
                throw new Exception("SimpleRNNLayer: input is null.");
            else if (!(input is Data2D))
                throw new Exception("SimpleRNNLayer: input is not Data2D.");

            this.input = input as Data2D;

            Dimension dimI = this.input.GetDimension();

            if (dimI.c != kernel.GetDimension().w)
            {
                throw new Exception("SimpleRNNLayer: kernel dimension is not equal with input dimension.");
            }

            int outputH = 1;
            int outputW = 1;
            int outputC = units;
            int outputB = dimI.b;

            output = new Data2D(outputH, outputW, outputC, outputB);
        }

        public void SetWeights(IData parameters)
        {
            if (parameters == null)
                throw new Exception("SimpleRNNLayer: parameters is null.");
            else if (!(parameters is Data2D))
                throw new Exception("SimpleRNNLayer: parameters is not Data2D.");

            Data2D pms = parameters as Data2D;

            if (pms.GetDimension().b != 3)
            {
                throw new Exception("SimpleRNNLayer: paramters should have 3 of batch size.");
            }

            if (pms.GetDimension().c != units)
            {
                throw new Exception("SimpleRNNLayer: paramters should have channel size with units number.");
            }

            if (pms.GetDimension().w != Math.Max(units, inputDim))
            {
                throw new Exception("SimpleRNNLayer: paramters has improper width size.");
            }

            kernel = new Data2D(1, inputDim, units, 1);
            recurrentKernel = new Data2D(1, units, units, 1);
            bias = new Data2D(1, 1, units, 1);

            Copy(kernel, pms, 0);
            Copy(recurrentKernel, pms, 1);
            Copy(bias, pms, 2);
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

        private void Copy(Data2D left, Data2D right, int batch)
        {
            for (int x = 0; x < left.GetDimension().w; ++x)
            {
                for (int y = 0; y < left.GetDimension().c; ++y)
                {
                    left[0, x, y, 0] = right[0, x, y, batch];
                }
            }
        }

        private int units;
        private int inputDim;
    }
}
