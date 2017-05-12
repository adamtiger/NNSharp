using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.KernelDescriptors
{
    public class Convolution1D : IKernelDescriptor
    {
        public Convolution1D(int padding, int stride, int kernelSize, int kernelNum)
        {
            this.padding = padding;
            this.stride = stride;
            this.kernelSize = kernelSize;
            this.kernelNum = kernelNum;

        }

        public int Padding { get { return padding; } }
        public int Stride { get { return stride; } }

        public int KernelSize { get { return kernelSize; } }
        public int KernelNum { get { return kernelNum; } }


        private int padding;
        private int stride;
        private int kernelSize;
        private int kernelNum;
    }
}
