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
            SigmoidLambda(input);
            output = input;
        }

        public static void SigmoidLambda(IData data)
        {
            data.ApplyToAll(p =>
            {
                return 1.0 / (1.0 + Math.Exp(-p));
            });
        }

        protected IData input;
        protected IData output;
    }
}
