using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NNSharp.DataTypes;
using NNSharp.Kernels.CPUKernels;

namespace NNSharp.SequentialBased.SequentialLayers
{
    [Serializable()]
    public class ELuLayer : ELuKernel, ILayer
    {

        public ELuLayer(double alpha)
        {
            this.alpha = alpha;
        }

        public IData GetOutput()
        {
            return output;
        }

        public void SetInput(IData input)
        {
            this.input = input;
        }

        public void SetWeights(IData weights)
        {
            // No weights.
        }
    }
}
