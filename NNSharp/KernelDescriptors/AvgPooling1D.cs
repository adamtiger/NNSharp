using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.KernelDescriptors
{
    public class AvgPooling1D : IKernelDescriptor
    {
        public AvgPooling1D(int padding, int stride, int kernelSize)
        {
            this.padding = padding;
            this.stride = stride;
            this.kernelSize = kernelSize;
        }

        public int Padding { get { return padding; } }
        public int Stride { get { return stride; } }
        public int KernelSize { get { return kernelSize; } }

        private int padding;
        private int stride;
        private int kernelSize;
    }
}
