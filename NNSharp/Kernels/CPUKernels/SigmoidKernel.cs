using NNSharp.DataTypes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.Kernels.CPUKernels
{
    [Serializable()]
    public class SigmoidKernel : IKernel
    {
        public void Execute()
        {
            input.ApplyToAll(p => { return 1.0 / (1.0 + Math.Exp(-p)); });
            output = input;
        }

        protected IData input;
        protected IData output;
    }
}
