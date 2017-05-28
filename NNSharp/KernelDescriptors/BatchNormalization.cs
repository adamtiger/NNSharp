using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.KernelDescriptors
{
    public class BatchNormalization : IKernelDescriptor
    {
        public BatchNormalization(double epsilon)
        {
            this.epsilon = epsilon;
        }

        public double Epsilon { get { return epsilon; } }

        private double epsilon;
    }
}
