using NNSharp.DataTypes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.Kernels.CPUKernels
{
    [Serializable()]
    public class Cropping1DKernel : IKernel
    {
        public void Execute()
        {
            for (int l = trimBegin; l < input.GetDimension().w - trimEnd; ++l)
            {
                for (int c = 0; c < input.GetDimension().c; ++c)
                {
                    for (int b = 0; b < input.GetDimension().b; ++b)
                    {
                        output[0, l - trimBegin, c, b] = input[0, l, c, b];
                    }
                }
            }
        }

        protected Data2D input;
        protected Data2D output;

        protected int trimBegin;
        protected int trimEnd;
    }
}
