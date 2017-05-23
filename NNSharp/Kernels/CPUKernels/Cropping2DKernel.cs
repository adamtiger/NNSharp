using NNSharp.DataTypes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.Kernels.CPUKernels
{
    [Serializable()]
    public class Cropping2DKernel : IKernel
    {
        public void Execute()
        {
            for (int h = topTrim; h < input.GetDimension().h - bottomTrim; ++h)
            {
                for (int w = leftTrim; w < input.GetDimension().w - rightTrim; ++w)
                {
                    for (int c = 0; c < input.GetDimension().c; ++c)
                    {
                        for (int b = 0; b < input.GetDimension().b; ++b)
                        {
                            output[h - topTrim, w - leftTrim, c, b] = input[h, w, c, b];
                        }
                    }
                }
            }
        }

        protected Data2D input;
        protected Data2D output;

        protected int topTrim;
        protected int bottomTrim;
        protected int leftTrim;
        protected int rightTrim;
    }
}
