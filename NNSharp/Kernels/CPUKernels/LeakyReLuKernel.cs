using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NNSharp.DataTypes;

namespace NNSharp.Kernels.CPUKernels
{
    [Serializable()]
    public class LeakyReLuKernel :IKernel
    {
        public void Execute()
        {
            LeakyReLuLambda(input);
            output = input;
        }

        public static void LeakyReLuLambda(IData data)
        {
            data.ApplyToAll(p =>
            {
                if (p >= 0.0)
                    return p;
                else
                {
                    return 0.3*p;
                }
            });
        }

        protected IData input;
        protected IData output;

        protected double alpha;
    }
}
