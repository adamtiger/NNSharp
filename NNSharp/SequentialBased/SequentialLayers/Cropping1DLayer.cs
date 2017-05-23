using NNSharp.Kernels.CPUKernels;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NNSharp.DataTypes;
using static NNSharp.DataTypes.SequentialModelData;
using static NNSharp.DataTypes.Data2D;

namespace NNSharp.SequentialBased.SequentialLayers
{
    [Serializable()]
    public class Cropping1DLayer : Cropping1DKernel, ILayer
    {
        public Cropping1DLayer(int trimBegin, int trimEnd)
        {
            this.trimBegin = trimBegin;
            this.trimEnd = trimEnd;
        }

        public IData GetOutput()
        {
            return output;
        }

        public void SetInput(IData input)
        {
            if (input == null)
                throw new Exception("Cropping1DLayer: input is null.");
            else if (!(input is Data2D))
                throw new Exception("Cropping1DLayer: input is not Data2D.");

            this.input = input as Data2D;
            Dimension dimI = this.input.GetDimension();

            if (dimI.h != 1)
            {
                throw new Exception("Cropping1DLayer: Input should be one dimensional.");
            }

            if (dimI.w - (trimBegin + trimEnd) < 1)
            {
                throw new Exception("Cropping1DLayer: Invalid cropping. More elements would be trimmed then exists.");
            }

            if (trimBegin < 0 || trimEnd < 0)
            {
                throw new Exception("Cropping1DLayer: Invalid cropping. Negative values for trimming.");
            }

            output = new Data2D(1, dimI.w - (trimBegin + trimEnd), dimI.c, dimI.b);
        }

        public void SetWeights(IData weights)
        {
            // No weights.
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
    }
}
