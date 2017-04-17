using NNSharp.DataTypes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.KernelDescriptors
{
    public class Convolution2D : IKernelDescriptor
    {
        public Convolution2D(int paddingVertical, int paddingHorizontal,
                             int strideVertical, int strideHorizontal,
                             int kernelHeight, int kernelWidth, int kernelNum)
        {
            this.paddingVertical = paddingVertical;
            this.paddingHorizontal = paddingHorizontal;
            this.strideVertical = strideVertical;
            this.strideHorizontal = strideHorizontal;

            this.kernelHeight = kernelHeight;
            this.kernelWidth = kernelWidth;
            this.kernelNum = kernelNum;

        }

        public int PaddingVertical { get { return paddingVertical; } }
        public int PaddingHorizontal { get { return paddingHorizontal; } }
        public int StrideVertical { get { return strideVertical; } }
        public int StrideHorizontal { get { return strideHorizontal; } }

        public int KernelHeight { get { return kernelHeight; } }
        public int KernelWidth { get { return kernelWidth; } }
        public int KernelNum { get { return kernelNum; } }


        private int paddingVertical;
        private int paddingHorizontal;
        private int strideVertical;
        private int strideHorizontal;

        private int kernelHeight;
        private int kernelWidth;
        private int kernelNum;
    }
}
