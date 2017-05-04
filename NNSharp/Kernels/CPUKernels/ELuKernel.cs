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
            input.ApplyToAll(p => {

                if (p >= 0.0)
                    return p;
                else
                {
                    return alpha * (Math.Exp(p) - 1.0);
                }
            });

            output = input;
        }

        protected IData input;
        protected IData output;

        protected double alpha;
    }
}
