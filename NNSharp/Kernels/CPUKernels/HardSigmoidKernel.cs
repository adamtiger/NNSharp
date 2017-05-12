using NNSharp.DataTypes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.Kernels.CPUKernels
{
    [Serializable()]
    public class HardSigmoidKernel : IKernel
    {
        public void Execute()
        {
            input.ApplyToAll(p => { return Math.Max(0.0, Math.Min(1, (p+1)/2)); });
            output = input;
        }

        protected IData input;
        protected IData output;
    }
}
