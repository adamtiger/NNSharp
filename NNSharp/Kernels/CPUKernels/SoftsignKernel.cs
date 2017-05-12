using NNSharp.DataTypes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.Kernels.CPUKernels
{
    [Serializable()]
    public class SoftsignKernel : IKernel
    {
        public void Execute()
        {
            input.ApplyToAll(p => { return p / (1 + Math.Abs(p)); });
            output = input;
        }
        protected IData input;
        protected IData output;
    }
}
