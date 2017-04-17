using NNSharp.DataTypes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static NNSharp.DataTypes.Data2D;

namespace NNSharp.Kernels.CPUKernels
{
    public class Bias2DKernel : IKernel
    {
        public void Execute()
        {
            Dimension dim = input.GetDimension();

            for (int b = 0; b < dim.b; ++b)
            {
                for (int c = 0; c < dim.c; ++c)
                {
                    for (int h = 0; h < dim.h; ++h)
                    {
                        for (int w = 0; w < dim.w; ++w)
                        {
                            input[h, w, c, b] += biases[c]; 
                        }
                    }
                }
            }
        }

        protected Data2D input;
        protected DataArray biases;
    }
}
