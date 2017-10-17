using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NNSharp.DataTypes;
using NNSharp.Kernels.CPUKernels;

namespace NNSharp.SequentialBased.SequentialLayers
{
    public class DropoutLayer : DropoutKernel, ILayer
    {

        public DropoutLayer(double rate, Data2D noiseShape)
        {
            this.rate = rate;
            this.noiseShape = noiseShape;
        }

        public IData GetOutput()
        {
            return output;
        }

        public void SetInput(IData input)
        {
            this.input = input as Data2D;
        }

        public void SetWeights(IData weights)
        {
            // no weights
        }

        public SequentialModelData.LayerData GetLayerSummary()
        {
            throw new NotImplementedException();
        }
    }
}
