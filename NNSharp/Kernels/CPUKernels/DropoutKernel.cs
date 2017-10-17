using NNSharp.DataTypes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.Kernels.CPUKernels
{
    public class DropoutKernel : IKernel
    {
        public void Execute()
        {
            // does nothing in the forward direction
        }

        protected Data2D input;
        protected Data2D output;

        protected double rate;
        protected Data2D noiseShape;
    }
}
