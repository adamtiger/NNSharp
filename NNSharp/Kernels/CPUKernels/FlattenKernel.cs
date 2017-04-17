using NNSharp.DataTypes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static NNSharp.DataTypes.Data2D;

namespace NNSharp.Kernels.CPUKernels
{
    public class FlattenKernel : IKernel
    {
        public void Execute()
        {
            int idx = 0;
            Dimension dim = input.GetDimension();

            for (int row = 0; row < dim.h; ++row)
            {
                for (int col = 0; col < dim.w; ++col)
                {
                    for (int chnl = 0; chnl < dim.c; ++chnl)
                    {
                        for (int batch = 0; batch < dim.b; ++batch)
                        {
                            output[0, 0, idx, batch] = input[row, col, chnl, batch];
                        }
                        idx += 1;
                    }
                }
            }
        }

        protected Data2D input;
        protected Data2D output;
    }
}
