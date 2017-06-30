using NNSharp.Kernels.CPUKernels;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NNSharp.DataTypes;
using static NNSharp.DataTypes.SequentialModelData;
using static NNSharp.DataTypes.Data2D;
using NNSharp.Kernels;

namespace NNSharp.SequentialBased.SequentialLayers
{
    [Serializable()]
    public class LSTMLayer : LSTMKernel, ILayer
    {
        public LSTMLayer(int units, int inputDim, ActivationLambda activation,
            ActivationLambda recurrentActivation)
        {
            this.units = units;
            this.inputDim = inputDim;
            this.activation = activation;
            this.recurrentActivation = recurrentActivation;

            h = new Data2D(1, 1, units, 1);
            C = new Data2D(1, 1, units, 1);
            C0 = new Data2D(1, 1, units, 1);
            i = new Data2D(1, 1, units, 1);
            f = new Data2D(1, 1, units, 1);
        }

        public IData GetOutput()
        {
            return output;
        }

        public void SetInput(IData input)
        {
            if (input == null)
                throw new Exception("LSTMLayer: input is null.");
            else if (!(input is Data2D))
                throw new Exception("LSTMLayer: input is not Data2D.");

            this.input = input as Data2D;

            Dimension dimI = this.input.GetDimension();

            if (units != wI.GetDimension().h)
            {
                throw new Exception("LSTMLayer: kernel height is not equal with units.");
            }

            if (dimI.c != wI.GetDimension().w)
            {
                throw new Exception("LSTMLayer: kernel width is not equal with input dim.");
            }

            if (units != uI.GetDimension().h)
            {
                throw new Exception("LSTMLayer: recurrent kernel height is not equal with units.");
            }

            if (units != uI.GetDimension().w)
            {
                throw new Exception("LSTMLayer: recurrent kernel width is not equal with units.");
            }

            if (units != bI.GetDimension().c)
            {
                throw new Exception("LSTMLayer: bias length is not equal with units.");
            }

            if (wI.GetDimension().Equals(wF.GetDimension()) && 
                wI.GetDimension().Equals(wO.GetDimension()) &&
                wI.GetDimension().Equals(wC.GetDimension()))
            {
                throw new Exception("LSTMLayer: kernel dimensions are not equal.");
            }

            if (uI.GetDimension().Equals(uF.GetDimension()) &&
                uI.GetDimension().Equals(uO.GetDimension()) &&
                uI.GetDimension().Equals(uC.GetDimension()))
            {
                throw new Exception("LSTMLayer: recurrent kernel dimensions are not equal.");
            }

            if (bI.GetDimension().Equals(bF.GetDimension()) &&
                bI.GetDimension().Equals(bO.GetDimension()) &&
                bI.GetDimension().Equals(bC.GetDimension()))
            {
                throw new Exception("LSTMLayer: bias dimensions are not equal.");
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
                throw new Exception("LSTMLayer: parameters is null.");
            else if (!(parameters is Data2D))
                throw new Exception("LSTMLayer: parameters is not Data2D.");

            Data2D pms = parameters as Data2D;

            if (pms.GetDimension().b != 12)
            {
                throw new Exception("LSTMRNNLayer: paramters should have 3 of batch size.");
            }

            if (pms.GetDimension().c != units)
            {
                throw new Exception("LSTMLayer: paramters should have channel size with units number.");
            }

            if (pms.GetDimension().w != Math.Max(units, inputDim))
            {
                throw new Exception("LSTMLayer: paramters has improper width size.");
            }

            if (pms.GetDimension().h != units)
            {
                throw new Exception("LSTMLayer: paramters has improper width size.");
            }

            wI = new Data2D(units, inputDim, 1, 1);
            wF = new Data2D(units, inputDim, 1, 1);
            wC = new Data2D(units, inputDim, 1, 1);
            wO = new Data2D(units, inputDim, 1, 1);

            uI = new Data2D(units, units, 1, 1);
            uF = new Data2D(units, units, 1, 1);
            uC = new Data2D(units, units, 1, 1);
            uO = new Data2D(units, units, 1, 1);

            bI = new Data2D(1, 1, units, 1);
            bF = new Data2D(1, 1, units, 1);
            bC = new Data2D(1, 1, units, 1);
            bO = new Data2D(1, 1, units, 1);

            Copy(wI, pms, 0);
            Copy(wF, pms, 1);
            Copy(wC, pms, 2);
            Copy(wO, pms, 3);

            Copy(uI, pms, 4);
            Copy(uF, pms, 5);
            Copy(uC, pms, 6);
            Copy(uO, pms, 7);

            Copy(bI, pms, 8);
            Copy(bF, pms, 9);
            Copy(bC, pms, 10);
            Copy(bO, pms, 11);
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
