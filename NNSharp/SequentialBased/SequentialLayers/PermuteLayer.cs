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
    public class PermuteLayer : PermuteKernel, ILayer
    {

        public PermuteLayer(int dim1, int dim2, int dim3)
        {
            this.dim1 = dim1;
            this.dim2 = dim2;
            this.dim3 = dim3;
        }

        public IData GetOutput()
        {
            return output;
        }

        public void SetInput(IData input)
        {
            if (input == null)
                throw new Exception("PermuteLayer: input is null.");
            else if (!(input is Data2D))
                throw new Exception("PermuteLayer: input is not Data2D.");

            this.input = input as Data2D;
            Dimension dimI = this.input.GetDimension();

            int[] sizes = { dimI.h, dimI.w, dimI.c};

            if (dim1 < 1 || dim1 > 3)
            {
                throw new Exception("PermuteLayer: The dim1 is not inside the right range: " + dim1);
            }

            if (dim2 < 1 || dim2 > 3)
            {
                throw new Exception("PermuteLayer: The dim2 is not inside the right range: " + dim2);
            }

            if (dim3 < 1 || dim3 > 3)
            {
                throw new Exception("PermuteLayer: The dim3 is not inside the right range: " + dim3);
            }

            if (dim1 == dim2 || dim1 == dim3 || dim2 == dim3)
            {
                throw new Exception("PermuteLayer: dim1, dim2, dim3 are incorrent. Some of them are equal!");
            }

            output = new Data2D(sizes[dim1 - 1], sizes[dim2 - 1], sizes[dim3 - 1], dimI.b);
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
