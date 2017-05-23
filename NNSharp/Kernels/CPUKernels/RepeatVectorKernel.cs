using NNSharp.DataTypes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.Kernels.CPUKernels
{
    public class RepeatVectorKernel : IKernel
    {
        public void Execute()
        {
            for (int batch = 0; batch < input.GetDimension().b; ++batch)
            {
                for (int repeat = 0; repeat < num; ++repeat)
                {
                    for (int channel = 0; channel < input.GetDimension().c; ++channel)
                    {
                        output[0, repeat, channel, batch] = input[0, 0, channel, batch];
                    }
                }
            }
        }

        protected Data2D input;
        protected Data2D output;
        protected int num;
    }
}
