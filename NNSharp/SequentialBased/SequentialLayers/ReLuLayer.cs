using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NNSharp.Kernels.CPUKernels;
using NNSharp.DataTypes;

namespace NNSharp.SequentialBased.SequentialLayers
{
    public class ReLuLayer : ReLuKernel, ILayer
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
