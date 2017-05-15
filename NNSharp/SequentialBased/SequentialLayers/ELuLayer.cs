using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NNSharp.DataTypes;
using NNSharp.Kernels.CPUKernels;
using static NNSharp.DataTypes.SequentialModelData;
using static NNSharp.DataTypes.Data2D;

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
