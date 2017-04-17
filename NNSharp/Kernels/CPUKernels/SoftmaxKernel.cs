using NNSharp.DataTypes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.Kernels.CPUKernels
{
    public class SoftmaxKernel : IKernel
    {
        public void Execute()
        {
            for (int b = 0; b < data.GetDimension().b; ++b)
            {
                double sum = 0.0;
                for (int i = 0; i < data.GetDimension().c; ++i)
                {
                    sum += Math.Exp(data[0, 0, i, b]);
                }

                for (int i = 0; i < data.GetDimension().c; ++i)
                {
                    data[0, 0, i, b] = Math.Exp(data[0, 0, i, b]) / sum;
                }
            }
        }

        protected Data2D data;
    }
}
