using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.KernelDescriptors
{
    public class LeakyReLu : IKernelDescriptor
    {

        public LeakyReLu(double alpha)
        {
            this.alpha = alpha;
        }

        public double Alpha { get { return alpha; } }
        private double alpha;
    }
}
