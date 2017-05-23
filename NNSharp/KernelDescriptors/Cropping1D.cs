using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.KernelDescriptors
{
    public class Cropping1D : IKernelDescriptor
    {
        public Cropping1D(int trimBegin, int trimEnd)
        {
            this.trimBegin = trimBegin;
            this.trimEnd = trimEnd;
        }

        public int TrimBegin { get { return trimBegin; } }
        public int TrimEnd { get { return trimEnd; } }

        private int trimBegin;
        private int trimEnd;
    }
}
