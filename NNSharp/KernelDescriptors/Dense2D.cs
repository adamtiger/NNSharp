using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.KernelDescriptors
{
    public class Dense2D : IKernelDescriptor
    {
        public Dense2D(int units)
        {
            this.units = units;
        }

        public int Units { get { return units; } }

        private int units;
    }
}
