using NNSharp.DataTypes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.Kernels.CPUKernels
{
    [Serializable()]
    public class ReLuKernel : IKernel
    {
        public void Execute()
        {
            ReLuLambda(input);
            output = input;
        }

        public static void ReLuLambda(IData data)
        {
            data.ApplyToAll(p =>
            {
                return Math.Max(0.0, p);
            });
        }

        protected IData input;
        protected IData output;
    }
}
