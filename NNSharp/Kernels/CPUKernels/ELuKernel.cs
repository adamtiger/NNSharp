using NNSharp.DataTypes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.Kernels.CPUKernels
{
    [Serializable()]
    public class ELuKernel : IKernel
    {
        public void Execute()
        {
            ELuLambda(input);

            output = input;
        }

        public static void ELuLambda(IData data)
        {
            data.ApplyToAll( p =>
            {
                if (p >= 0.0)
                    return p;
                else
                {
                    return 1.0 * (Math.Exp(p) - 1.0);
                }
            });
        }

        protected IData input;
        protected IData output;

        protected double alpha;
    }
}
