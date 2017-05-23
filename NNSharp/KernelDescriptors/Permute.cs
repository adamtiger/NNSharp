using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.KernelDescriptors
{
    public class Permute : IKernelDescriptor
    {
        // Batch remains unchanged.
        public Permute(int dim1, int dim2, int dim3)
        {

        }

        public int Dim1 { get { return dim1; } }
        public int Dim2 { get { return dim1; } }
        public int Dim3 { get { return dim1; } }

        private int dim1;
        private int dim2;
        private int dim3;
    }
}
