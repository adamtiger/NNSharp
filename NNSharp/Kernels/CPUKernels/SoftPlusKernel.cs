using NNSharp.DataTypes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.Kernels.CPUKernels
{
    [Serializable()]
    public class SoftPlusKernel : IKernel
    {
        public void Execute()
        {
            input.ApplyToAll(p => { return Math.Log(1 + Math.Exp(p)); });
            output = input;
        }
        protected IData input;
        protected IData output;
    }
}
