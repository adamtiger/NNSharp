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
    public class RepeatVectorLayer : RepeatVectorKernel, ILayer
    {
        public RepeatVectorLayer(int num)
        {
            this.num = num;
        }

        public IData GetOutput()
        {
            return output;
        }

        public void SetInput(IData input)
        {
            if (input == null)
                throw new Exception("RepeatVectorLayer: input is null.");
            else if (!(input is Data2D))
                throw new Exception("RepeatVectorLayer: input is not Data2D.");

            this.input = input as Data2D;
            Dimension dimI = this.input.GetDimension();

            if (dimI.h != 1)
            {
                throw new Exception("RepeatVectorLayer: The input height should be 1.");
            }

            if (dimI.w != 1)
            {
                throw new Exception("RepeatVectorLayer: The input width should be 1.");
            }

            output = new Data2D(1, num, dimI.c, dimI.b);
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
