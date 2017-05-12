using NNSharp.DataTypes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static NNSharp.DataTypes.Data2D;

namespace NNSharp.Kernels.CPUKernels
{
    [Serializable()]
    public class Conv1DKernel : IKernel
    {
        public void Execute()
        {
            Dimension dimK = weights.GetDimension();
            Dimension dimI = input.GetDimension();
            Dimension dimO = output.GetDimension();
            int stH = 0;

            for (int batch = 0; batch < dimO.b; ++batch)
            {
                for (int filter = 0; filter < dimO.c; ++filter)
                {
                    for (int l = 0; l < dimO.w; ++l)
                    {

                        stH = l * stride - padding;
                        output[0, l, filter, batch] = 0.0;

                        for (int idx = stH; idx < stH + dimK.w; ++idx)
                        {
                            for (int idxC = 0; idxC < dimK.c; ++idxC)
                            {
                                output[0, l, filter, batch] += input[0, idx, idxC, batch] *
                                                                    weights[0, idx - stH, idxC, filter];
                            }
                        }
                    }
                }
            }
        }

        protected Data2D input;
        protected Data2D output;
        protected Data2D weights;

        protected int padding;
        protected int stride;
    }
}
