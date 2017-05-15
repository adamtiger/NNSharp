using NNSharp.DataTypes;
using NNSharp.Kernels.CPUKernels;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static NNSharp.DataTypes.SequentialModelData;

namespace NNSharp.SequentialBased.SequentialLayers
{
    [Serializable()]
    public class SigmoidLayer : SigmoidKernel, ILayer
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

        public LayerData GetLayerSummary()
        {
            // The input and the output have the same sizes as the output
            // of the previous layer.
            return new LayerData(
                this.ToString(),
                -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1);
        }
    }
}
