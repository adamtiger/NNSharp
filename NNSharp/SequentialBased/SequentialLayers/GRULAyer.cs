using NNSharp.Kernels.CPUKernels;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NNSharp.DataTypes;
using NNSharp.Kernels;
using static NNSharp.DataTypes.Data2D;
using static NNSharp.DataTypes.SequentialModelData;

namespace NNSharp.SequentialBased.SequentialLayers
{
    [Serializable()]
    public class GRULayer : GRUKernel, ILayer
    {

        public GRULayer(int units, int inputDim, ActivationLambda activation,
            ActivationLambda recurrentActivation)
        {
            this.units = units;
            this.inputDim = inputDim;
            this.activation = activation;
            this.recurrentActivation = recurrentActivation;

            z = new Data2D(1, 1, units, 1);
            r = new Data2D(1, 1, units, 1);
            hh = new Data2D(1, 1, units, 1);
            h = new Data2D(1, 1, units, 1);
        }

        public IData GetOutput()
        {
            return output;
        }

        public void SetInput(IData input)
        {
            if (input == null)
                throw new Exception("GRULayer: input is null.");
            else if (!(input is Data2D))
                throw new Exception("GRULayer: input is not Data2D.");

            this.input = input as Data2D;

            Dimension dimI = this.input.GetDimension();

            int outputH = 1;
            int outputW = 1;
            int outputC = units;
            int outputB = dimI.b;

            output = new Data2D(outputH, outputW, outputC, outputB);
        }

        public void SetWeights(IData parameters)
        {
            if (parameters == null)
                throw new Exception("GRULayer: parameters is null.");
            else if (!(parameters is Data2D))
                throw new Exception("GRULayer: parameters is not Data2D.");

            Data2D pms = parameters as Data2D;

            if (pms.GetDimension().b != 9)
            {
                throw new Exception("GRULayer: paramters should have 3 of batch size.");
            }

            if (pms.GetDimension().c != units)
            {
                throw new Exception("GRULayer: paramters should have channel size with units number.");
            }

            if (pms.GetDimension().w != Math.Max(units, inputDim))
            {
                throw new Exception("GRULayer: paramters has improper width size.");
            }

            if (pms.GetDimension().h != units)
            {
                throw new Exception("GRULayer: paramters has improper width size.");
            }

            wZ = new Data2D(units, inputDim, 1, 1);
            wR = new Data2D(units, inputDim, 1, 1);
            wHH = new Data2D(units, inputDim, 1, 1);

            uZ = new Data2D(units, units, 1, 1);
            uR = new Data2D(units, units, 1, 1);
            uHH = new Data2D(units, units, 1, 1);

            bZ = new Data2D(1, 1, units, 1);
            bR = new Data2D(1, 1, units, 1);
            bHH = new Data2D(1, 1, units, 1);

            Copy(wZ, pms, 0);
            Copy(wR, pms, 1);
            Copy(wHH, pms, 2);

            Copy(uZ, pms, 3);
            Copy(uR, pms, 4);
            Copy(uHH, pms, 5);

            Copy(bZ, pms, 6);
            Copy(bR, pms, 7);
            Copy(bHH, pms, 8);
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
            for (int x = 0; x < left.GetDimension().h; ++x)
            {
                for (int y = 0; y < left.GetDimension().w; ++y)
                {
                    for (int z = 0; z < left.GetDimension().c; ++z)
                    {
                        left[x, y, z, 0] = right[x, y, z, batch];
                    }
                }
            }
        }

        private int units;
        private int inputDim;
    }
}
