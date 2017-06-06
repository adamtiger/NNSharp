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
            SoftsignLambda(input);
            output = input;
        }

        public static void SoftsignLambda(IData data)
        {
            data.ApplyToAll(p =>
            {
                return p / (1 + Math.Abs(p));
            });
        }

        protected IData input;
        protected IData output;
    }
}
