using NNSharp.DataTypes;
using NNSharp.KernelDescriptors;
using NNSharp.Kernels.CPUKernels;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.SequentialBased.SequentialLayers
{
    [Serializable()]
    public class SoftsignLayer : SoftsignKernel, ILayer
    {

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
