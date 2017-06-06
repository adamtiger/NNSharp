using NNSharp.DataTypes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.Kernels.CPUKernels
{

    [Serializable()]
    public class TanHKernel : IKernel
    {
        public void Execute()
        {
            TanHLambda(input);
            output = input;
        }

        public static void TanHLambda(IData data)
        {
            data.ApplyToAll(p =>
            {
                return 2.0 / (1 + Math.Exp(-2.0 * p)) - 1.0;
            });
        }

        protected IData input;
        protected IData output;
    }
}
