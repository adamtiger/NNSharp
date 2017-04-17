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
    public class FlattenLayer : FlattenKernel, ILayer
    {

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

            Dimension dim = this.input.GetDimension();

            int chnlSize = dim.h * dim.w * dim.c;
            int batchSize = dim.b;

            output = new Data2D(1, 1, chnlSize, batchSize);
        }

        public void SetWeights(IData weights)
        {
            // No weights.
        }
    }
}
