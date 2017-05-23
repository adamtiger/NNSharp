using NNSharp.DataTypes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.Kernels.CPUKernels
{
    public class PermuteKernel : IKernel
    {
        public void Execute()
        {
            int[] idx = { 0, 0, 0 };
            for (int h = 0; h < input.GetDimension().h; ++h)
            {
                for (int w = 0; w < input.GetDimension().w; ++w)
                {
                    for (int c = 0; c < input.GetDimension().c; ++c)
                    {
                        idx[0] = h; idx[1] = w; idx[2] = c;

                        for (int b = 0; b < input.GetDimension().b; ++b) {

                            output[idx[dim1 - 1], idx[dim2 - 1], idx[dim3 - 1], b] = input[h, w, c, b];
                        }
                    }
                }
            }
        }

        protected Data2D input;
        protected Data2D output;

        protected int dim1, dim2, dim3;
    }
}
