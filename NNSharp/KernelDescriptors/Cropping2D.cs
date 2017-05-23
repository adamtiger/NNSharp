using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.KernelDescriptors
{
    public class Cropping2D : IKernelDescriptor
    {
        public Cropping2D(int topTrim, int bottomTrim, int leftTrim, int rightTrim)
        {
            this.topTrim = topTrim;
            this.bottomTrim = bottomTrim;
            this.leftTrim = leftTrim;
            this.rightTrim = rightTrim;
        }

        public int TopTrim { get { return topTrim; } }
        public int BottomTrim { get { return bottomTrim; } }
        public int LeftTrim { get { return leftTrim; } }
        public int RightTrim { get { return rightTrim; } }

        private int topTrim;
        private int bottomTrim;
        private int leftTrim;
        private int rightTrim;
    }
}
