using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.KernelDescriptors
{
    public class RepeatVector : IKernelDescriptor
    {
        public RepeatVector(int num)
        {
            this.num = num;
        }

        public int Num { get { return num; } }

        private int num;
    }
}
