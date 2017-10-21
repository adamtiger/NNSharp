using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NNSharp.DataTypes;
using NNSharp.Kernels.CPUKernels;
using static NNSharp.DataTypes.Data2D;
using static NNSharp.DataTypes.SequentialModelData;

namespace NNSharp.SequentialBased.SequentialLayers
{
    [Serializable()]
    public class Dense2DLayer : Dense2DKernel, ILayer
    {

        public Dense2DLayer(int units)
        {
            this.units = units;
        }

        public IData GetOutput()
        {
            return output;
        }

        public void SetInput(IData input)
        {
            if (input == null)
                throw new Exception("Dense2DLayer: input is null.");
            else if (!(input is Data2D))
                throw new Exception("Dense2DLayer: input is not Data2D.");

            this.input = input as Data2D;

            Dimension dimI = this.input.GetDimension();
            Dimension dimK = this.weights.GetDimension();

            if (dimI.c != dimK.c)
                throw new Exception("Wrong kernel and input sizes: sizes of channels should match." +
                   " Now: dimI.c: " + dimI.c + " != dimK.c: " + dimK.c);

            if (dimI.h != dimK.h)
                throw new Exception("Wrong kernel and input sizes: sizes of heights should match." +
                   " Now: dimI.h: " + dimI.h + " != dimK.h: " + dimK.h);

            if (dimI.w != dimK.w)
                throw new Exception("Wrong kernel and input sizes: sizes of widths should match." +
                   " Now: dimI.w: " + dimI.w + " != dimK.w: " + dimK.w);

            output = new Data2D(1, 1, dimK.b, dimI.b);
        }

        public void SetWeights(IData weights)
        {
            if (weights == null)
                throw new Exception("Dense2DLayer: weights is null.");
            else if (!(weights is Data2D))
                throw new Exception("Dense2DLayer: weights is not Data2D.");
            else if (((Data2D)weights).GetDimension().b != units)
                throw new Exception("Dense2DLayer: batch size of weights is not appropriate.");

            this.weights = weights as Data2D;
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


        private int units;
    }
}
