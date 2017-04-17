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
    public class Dense2DLayer : Dense2DKernel, ILayer
    {

        public Dense2DLayer(IData weights)
        {
            if (weights == null)
                throw new Exception("Dense2DLayer: weights is null.");
            else if (!(weights is Data2D))
                throw new Exception("Dense2DLayer: weights is not Data2D.");

            this.weights = weights as Data2D;
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
                   " Now: dimI: " + dimI.c + " != dimK: " + dimK.c);

            if (dimI.h != dimK.h)
                throw new Exception("Wrong kernel and input sizes: sizes of heights should match." +
                   " Now: dimI: " + dimI.h + " != dimK: " + dimK.h);

            if (dimI.w != dimK.w)
                throw new Exception("Wrong kernel and input sizes: sizes of widths should match." +
                   " Now: dimI: " + dimI.w + " != dimK: " + dimK.w);

            output = new Data2D(1, 1, dimK.b, dimI.b);
        }

        public void SetWeights(IData weights)
        {
            if (weights == null)
                throw new Exception("Dense2DLayer: weights is null.");
            else if (!(weights is Data2D))
                throw new Exception("Dense2DLayer: weights is not Data2D.");

            this.weights = weights as Data2D;
        }
    }
}
