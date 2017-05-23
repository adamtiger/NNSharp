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
    public class Cropping2DLayer : Cropping2DKernel, ILayer
    {
        public Cropping2DLayer(int topTrim, int bottomTrim, int leftTrim, int rightTrim)
        {
            this.topTrim = topTrim;
            this.bottomTrim = bottomTrim;
            this.leftTrim = leftTrim;
            this.rightTrim = rightTrim;
        }

        public IData GetOutput()
        {
            return output;
        }

        public void SetInput(IData input)
        {
            if (input == null)
                throw new Exception("Cropping2DLayer: input is null.");
            else if (!(input is Data2D))
                throw new Exception("Cropping2DLayer: input is not Data2D.");

            this.input = input as Data2D;
            Dimension dimI = this.input.GetDimension();

            if (dimI.h - (topTrim + bottomTrim) < 1)
            {
                throw new Exception("Cropping2DLayer: Invalid cropping. More elements would be trimmed then exists.");
            }

            if (dimI.w - (leftTrim + rightTrim) < 1)
            {
                throw new Exception("Cropping2DLayer: Invalid cropping. More elements would be trimmed then exists.");
            }

            if (topTrim < 0 || bottomTrim < 0 || leftTrim < 0 || rightTrim < 0)
            {
                throw new Exception("Cropping2DLayer: Invalid cropping. Negative values for trimming.");
            }

            output = new Data2D(dimI.h - (topTrim + bottomTrim), dimI.w - (leftTrim + rightTrim), dimI.c, dimI.b);
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
