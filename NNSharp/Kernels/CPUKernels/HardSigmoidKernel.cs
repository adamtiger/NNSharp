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
            HardSigmoidLambda(input);
            output = input;
        }

        public static void HardSigmoidLambda(IData data)
        {
            data.ApplyToAll(p =>
            {
                return Math.Max(0.0, Math.Min(1, 0.2 * p + 0.5));
            });
        }

        protected IData input;
        protected IData output;
    }
}
